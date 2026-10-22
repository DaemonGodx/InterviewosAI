import logging
import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError
from groq import APIError
from app.config import GROQ_API_KEY
from app.models.schemas import (
    InterviewQuestion,
    AnswerAnalysis,
    AdaptiveDecision,
    EvaluationReport,
    TurnIntent,
)
from app.engine.state import GraphState

logger = logging.getLogger(__name__)

# Hidden last-resort safeguards. These are intentionally not interview settings:
# answer evidence determines normal progression; these caps only stop loops.
MAX_FOLLOW_UPS_PER_SOURCE = 3
MAX_FOLLOW_UPS_PER_INTERVIEW = 3
MAX_AMBIGUOUS_TURNS = 2
MAX_UNKNOWN_ANSWERS_PER_SOURCE = 2
MAX_HINTS_PER_QUESTION = 2
TURN_CONFIDENCE_THRESHOLD = 0.62


class AdaptiveDecisionSourceMismatch(ValueError):
    """Raised when a model assigns an adaptive decision to another question."""


def _safe_schema_error_context(error: ValidationError):
    """Return field paths and error codes only, never rejected input values."""
    return [
        {
            "field": ".".join(str(part) for part in item.get("loc", ())),
            "kind": str(item.get("type", "validation_error")),
        }
        for item in error.errors()[:8]
    ]


def _analysis_error_details(error: Exception):
    """Expose provider diagnostics without logging candidate/model content."""
    if isinstance(error, APIError):
        body = getattr(error, "body", None)
        try:
            body_text = json.dumps(body, ensure_ascii=False, default=str)
        except Exception:
            body_text = repr(body)
        # The response body is the provider's error payload, not the interview
        # prompt or candidate answer. Redact the configured credential defensively.
        if GROQ_API_KEY:
            body_text = body_text.replace(GROQ_API_KEY, "[REDACTED]")
        return {
            "status_code": getattr(error, "status_code", None),
            "request_id": getattr(error, "request_id", None),
            "provider_error_body": body_text,
        }
    if isinstance(error, ValidationError):
        return {"validation_errors": _safe_schema_error_context(error)}
    if isinstance(error, OutputParserException):
        # Parser exceptions can contain the full model completion, which may
        # echo the candidate's answer. Identify the failure without logging it.
        return {
            "parser_error": type(error).__name__,
            "parser_observation": getattr(error, "observation", None),
        }
    return {"error_message": str(error)}


def _explicit_requested_action(utterance: str):
    """Recover clear direct requests if the classifier omits its secondary field."""
    text = " ".join((utterance or "").casefold().split())
    if re.search(r"\b(?:can|could|may) i skip\b|\bskip (?:this )?(?:question|one)\b", text):
        return "SKIP"
    if re.search(r"\b(?:ask|give|make|switch|change|go)\b.{0,50}\b(?:easier|easy|simpler)\b|\b(?:easier|simpler) question\b", text):
        return "EASIER"
    if re.search(r"\b(?:ask|give|want|need|can|could|provide)\b.{0,35}\bfollow[ -]?up\b", text):
        return "FOLLOW_UP"
    if re.search(r"\b(?:give|offer|want|need|get|have|can|could)\b.{0,30}\b(?:hint|clue)\b|\b(?:hint|clue) please\b", text):
        return "HINT"
    if re.search(r"\b(?:repeat|restate|say|read)\b.{0,30}\b(?:question|that|again)\b", text):
        return "REPEAT"
    if re.search(r"\b(?:simplify|rephrase|clarify|explain)\b.{0,35}\b(?:question|wording|what you mean|that)\b", text):
        return "CLARIFY"
    return None


def _is_plain_admission(utterance: str):
    """Recognize an admission clause without mistaking technical uncertainty for one."""
    text = " ".join((utterance or "").casefold().split())
    return bool(re.match(
        r"^(?:i\s+)?(?:don't|dont|do not)\s+know(?:\s+this)?(?=$|[\s,;:.!?])|^(?:no idea|i have no idea|not sure)(?=$|[\s,;:.!?])",
        text,
    ))

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2
)


def classify_turn_node(state: GraphState):
    """Classify a candidate utterance before it can enter answer analysis."""
    utterance = (state.get("current_input") or "").strip()
    normalized = " ".join(utterance.casefold().strip(" .!?\t\n\"'").split())
    # The voice UI uses this exact sentinel when the candidate explicitly skips.
    explicit_request = _explicit_requested_action(utterance)
    plain_admission = _is_plain_admission(utterance)
    if normalized in {"no answer provided", "skip", "skip this question", "i want to skip"}:
        classification = TurnIntent(intent="SKIP", confidence=1.0)
    elif explicit_request == "EASIER" and re.fullmatch(
        r"(?:please\s+)?(?:go\s+(?:a\s+bit\s+)?(?:easy|easier)|"
        r"make\s+(?:it|this|the question)\s+(?:easy|easier|simpler)|"
        r"ask\s+(?:me\s+)?(?:an?\s+)?(?:easy|easier|simpler)(?:\s+question)?)"
        r"(?:\s+please)?",
        normalized,
    ):
        classification = TurnIntent(
            intent="ADMITTED_UNKNOWN", confidence=1.0, requested_action="EASIER",
        )
    # A few complete, unambiguous admissions are cheap to recognize locally.
    # Keep this an exact whole-utterance match: technical answers such as
    # "I don't know for sure, but I would inspect the logs" must still reach
    # the structured classifier and remain eligible for answer analysis.
    elif normalized in {
        "i don't know", "i dont know", "i do not know", "don't know", "dont know",
        "no idea", "i have no idea", "not sure",
    } or (plain_admission and explicit_request in {"EASIER", "FOLLOW_UP"}):
        # Preserve compound admission + request semantics even when the model
        # is unavailable or returns low confidence.
        classification = TurnIntent(
            intent="ADMITTED_UNKNOWN", confidence=1.0, requested_action=explicit_request,
        )
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Classify the candidate's latest utterance in an interview. Return one primary intent, confidence 0..1, and requested_action when the candidate explicitly asks for one. "
             "Intents: ANSWER for any attempt containing relevant evidence, reasoning, an example, or a technical claim, even if uncertain; "
             "ADMITTED_UNKNOWN only for a plain explicit admission such as 'I don't know' with no substantive answer; "
             "REPEAT, CLARIFY, HINT, SKIP, CORRECTION, OFF_TOPIC, or AMBIGUOUS. "
             "requested_action is independent of intent and may be EASIER (explicitly asks for an easier question), FOLLOW_UP (explicitly asks for a follow-up), HINT, CLARIFY, REPEAT, or SKIP; otherwise null. "
             "For compound utterances such as 'I don't know; ask me an easier question' or the equivalent non-native phrasing 'I don't know this, ask a bit easy' keep intent=ADMITTED_UNKNOWN and requested_action=EASIER. "
             "For 'I don't know; ask me a follow-up' use ADMITTED_UNKNOWN plus FOLLOW_UP; for 'I don't know; give me a hint' use ADMITTED_UNKNOWN plus HINT. "
             "A request to simplify or explain wording is CLARIFY, not EASIER. A request to repeat is REPEAT. Do not infer EASIER from an admission alone. "
             "A question about the wording, scope, or terminology of the interview question is CLARIFY, not OFF_TOPIC. "
             "Use OFF_TOPIC only for a request or comment unrelated to the interview question. "
             "A genuine answer that says 'I think', 'maybe', or expresses uncertainty remains ANSWER. "
             "Use AMBIGUOUS when intent is unclear. Current question: {question}. Utterance: {utterance}"),
            ("human", "Classify this utterance."),
        ])
        try:
            chain = prompt | llm.with_structured_output(TurnIntent)
            classification = TurnIntent.model_validate(chain.invoke({
                "question": state.get("current_question") or "Unavailable",
                "utterance": utterance,
            }))
            if classification.confidence < TURN_CONFIDENCE_THRESHOLD:
                classification = TurnIntent(
                    intent="AMBIGUOUS", confidence=classification.confidence,
                    requested_action=classification.requested_action,
                )
        except Exception:
            logger.warning("Turn intent classification failed; applying reliable local request handling.", exc_info=True)
            classification = TurnIntent(intent="AMBIGUOUS", confidence=0.0)

    if classification.requested_action is None and explicit_request:
        classification = classification.model_copy(update={"requested_action": explicit_request})
    if explicit_request == "EASIER" and classification.intent == "AMBIGUOUS":
        # A clear easier-question request implies the candidate cannot answer
        # at the current level. Promote it into the existing unknown-answer
        # path so the graph honors the request instead of treating it as vague.
        classification = classification.model_copy(update={"intent": "ADMITTED_UNKNOWN"})

    ambiguous_count = state.get("consecutive_ambiguous_turns", 0)
    if classification.intent == "AMBIGUOUS":
        ambiguous_count += 1
    else:
        ambiguous_count = 0
    return {
        "current_turn_intent": classification.model_dump(mode="json"),
        "current_input": None,
        "classified_utterance": utterance,
        "consecutive_ambiguous_turns": ambiguous_count,
    }


def respond_to_conversation_node(state: GraphState):
    """Respond to a conversational request without changing interview progress."""
    turn = state.get("current_turn_intent") or {}
    intent = turn.get("requested_action") or turn.get("intent", "AMBIGUOUS")
    if intent in {"EASIER", "FOLLOW_UP"} and turn.get("intent") not in {"ANSWER", "ADMITTED_UNKNOWN"}:
        intent = "AMBIGUOUS"
    if intent == "EASIER" and (state.get("current_question_difficulty") or state.get("current_difficulty")) == "Easy":
        # At the existing minimum difficulty, use the established hint path and
        # keep the current question active instead of inventing a new level.
        intent = "HINT"
    utterance = state.get("current_input") or ""
    # current_input is consumed by classification; use this value saved alongside
    # the intent in the transient field when creating the history record.
    utterance = state.get("classified_utterance") or utterance
    question = state.get("current_question") or "the current interview question"
    metadata = {
        "topic": state.get("current_question_topic") or "General",
        "question_type": state.get("current_question_type") or "Technical",
        "behavioral_category": state.get("current_behavioral_category"),
        "primary_competency": state.get("current_primary_competency") or "",
        "difficulty": state.get("current_question_difficulty") or state.get("current_difficulty") or "Medium",
        "expected_concepts": state.get("current_expected_concepts") or [],
    }
    fixed = {
        "REPEAT": question,
        "CORRECTION": "Go ahead and provide the corrected or completed answer to the same question.",
        "OFF_TOPIC": f"Let's stay focused on the interview. {question}",
    }
    response = fixed.get(intent)
    previous_turns = [
        item for item in state.get("conversation_history", [])
        if isinstance(item, dict) and item.get("question") == question
    ]
    hint_count = sum(item.get("intent") == "HINT" for item in previous_turns)
    previous_turns = previous_turns[-3:]
    ambiguous_count = state.get("consecutive_ambiguous_turns", 0)
    if intent == "AMBIGUOUS":
        response = (
            "Please choose whether you want to answer, get a hint, or skip this question."
            if ambiguous_count >= MAX_AMBIGUOUS_TURNS
            else "Would you like to answer, hear a hint, or skip this question?"
        )
    elif intent == "HINT" and hint_count >= MAX_HINTS_PER_QUESTION:
        response = "I can offer one more detail only by giving away the solution. Please try your approach or skip this question."
    elif intent in {"CLARIFY", "HINT"}:
        if intent == "CLARIFY":
            kind = (
                "Briefly rephrase or explain the wording, terminology, or scope. Preserve the assessed concept, "
                "topic, and competency. Do not turn the clarification into an answer or add new requirements."
            )
        elif hint_count == 0:
            kind = (
                "Give exactly one concise, conceptual direction as the first hint. Start with what kind of "
                "thing to consider, not steps, implementation, or a solution."
            )
        else:
            kind = (
                "Give exactly one slightly more specific second hint, grounded in at most one expected concept. "
                "Do not outline implementation steps or reveal the final result, ideal answer, or full solution."
            )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional, neutral, respectful technical interviewer. Be concise and conversational; "
             "do not overpraise, apologize unnecessarily, claim human identity, or invent personal experience. "
             "Follow this intent-specific instruction exactly: {instruction} "
             "Use only the supplied question context. Keep to at most two sentences and ask no extra question. "
             "Question: {question}; type: {question_type}; behavioral category: {behavioral_category}; "
             "topic: {topic}; competency: {competency}; difficulty: {difficulty}; expected concepts: {concepts}; "
             "prior conversation on this question: {prior_turns}"),
            ("human", "Intent: {intent}. Candidate's latest utterance: {utterance}"),
        ])
        try:
            response = (prompt | llm).invoke({
                "instruction": kind,
                "intent": intent,
                "question": question,
                "question_type": metadata["question_type"],
                "behavioral_category": metadata["behavioral_category"] or "Not applicable",
                "topic": metadata["topic"],
                "competency": metadata["primary_competency"] or metadata["topic"],
                "difficulty": metadata["difficulty"],
                "concepts": ", ".join(metadata["expected_concepts"]),
                "prior_turns": " | ".join(
                    f"{item.get('intent')}: {item.get('utterance')} -> {item.get('response')}"
                    for item in previous_turns
                ) or "None",
                "utterance": utterance,
            })
            response = getattr(response, "content", response)
            if not isinstance(response, str) or not response.strip():
                raise ValueError("Empty conversation response")
            response = response.strip()
        except Exception:
            concepts = metadata["expected_concepts"]
            hint_concept = concepts[min(hint_count, len(concepts) - 1)] if concepts else metadata["topic"]
            if intent == "HINT":
                response = (
                    f"Start by considering the {hint_concept} concept and one observation that could help you "
                    "reason about the question."
                    if hint_count == 0 else
                    f"Use the {hint_concept} concept to organize your reasoning; work through the remaining details "
                    "yourself."
                )
            else:
                response = f"The wording is asking you to focus on {metadata['topic']} and explain your reasoning within that scope."

    history = list(state.get("conversation_history", []))
    history_item = {"question": question, "utterance": utterance, "intent": intent, "response": response}
    if turn.get("requested_action"):
        history_item["requested_action"] = turn["requested_action"]
    if turn.get("intent") == "ADMITTED_UNKNOWN" and intent != "ADMITTED_UNKNOWN":
        history_item["underlying_intent"] = "ADMITTED_UNKNOWN"
    history.append(history_item)
    return {
        "interviewer_response": response,
        "conversation_history": history,
        "classified_utterance": None,
    }


def prepare_skip_node(state: GraphState):
    """Use the existing blank/no-evidence analysis path for an explicit skip."""
    turn = dict(state.get("current_turn_intent") or {})
    automatic = turn.get("intent") == "AMBIGUOUS"
    response = None
    if automatic:
        turn.update({"intent": "SKIP", "requested_action": "SKIP"})
        response = "We haven't found an answer yet, so I'll move on to the next question."
    history = list(state.get("conversation_history", []))
    item = {
        "utterance": state.get("classified_utterance") or "",
        "intent": "SKIP",
        "response": response or "Question skipped; no candidate score was assigned.",
    }
    if automatic:
        item["automatic"] = True
        item["reason"] = "ambiguous_turn_limit_reached"
    history.append(item)
    update = {
        "current_answer": "",
        "classified_utterance": None,
        "conversation_history": history,
        "consecutive_ambiguous_turns": 0,
    }
    if automatic:
        update["current_turn_intent"] = turn
        update["interviewer_response"] = response
    return update


def accept_answer_node(state: GraphState):
    """Move classified answer text into the existing analysis input field."""
    return {
        "current_answer": state.get("classified_utterance") or "",
        "classified_utterance": None,
    }

def generate_question_node(state: GraphState):
    """Generates the next question based on profile and state."""
    interview_plan = state.get('interview_plan', [])
    config = state.get('config') or {}
    configured_count = config.get('question_count', len(interview_plan))
    question_number = len(state.get('questions', [])) + 1
    pending = state.get('pending_adaptive_question')
    if interview_plan:
        plan_cursor = state.get('plan_cursor', 0)
        execution_order = state.get('plan_execution_order') or list(range(len(interview_plan)))
        if pending and pending.get('is_follow_up'):
            plan_index = pending.get('source_plan_index')
            if not isinstance(plan_index, int) or not 0 <= plan_index < len(interview_plan):
                logger.error("Invalid follow-up source index %r; skipping pending follow-up", plan_index)
                pending = None
            else:
                planned = interview_plan[plan_index]
                if hasattr(planned, "model_dump"):
                    planned = planned.model_dump()
                question_source = pending.get('question') or {}
                is_follow_up = True
                advance_cursor = False
        if not pending and plan_cursor >= len(execution_order):
            return {}
        elif not pending:
            plan_index = execution_order[plan_cursor]
            if plan_index < 0 or plan_index >= len(interview_plan):
                logger.error("Invalid plan execution index %s; stopping plan generation", plan_index)
                return {}
            planned = interview_plan[plan_index]
            if hasattr(planned, "model_dump"):
                planned = planned.model_dump()
            question_source = planned
            is_follow_up = False
            advance_cursor = True
        override = state.get('next_difficulty_override')
        question_meta = {
            "question": question_source["question"],
            "question_type": question_source["question_type"],
            "behavioral_category": question_source.get("behavioral_category") or planned.get("behavioral_category"),
            "topic": question_source.get("topic") or planned["topic"],
            "primary_competency": question_source.get("primary_competency") or question_source.get("topic") or planned["topic"],
            "planned_difficulty": planned["difficulty"],
            "difficulty": override or question_source.get("difficulty") or planned["difficulty"],
            "expected_concepts": question_source.get("expected_concepts") or planned["expected_concepts"],
            "plan_index": plan_index,
            "is_follow_up": is_follow_up,
            "source_plan_index": plan_index,
            "follow_up_for_plan_index": plan_index if is_follow_up else None,
            "adaptation_reason": pending.get('rationale') if is_follow_up and pending else None,
            "difficulty_override": override,
        }
        questions = state.get('questions', []) + [question_meta["question"]]
        topics = state.get('topics_covered', []) + [question_meta["topic"]]
        planned_asked = plan_cursor + (1 if advance_cursor else 0)
        remaining = max(0, configured_count - planned_asked)
        return {
            "current_question": question_meta["question"],
            "current_question_type": question_meta["question_type"],
            "current_behavioral_category": question_meta["behavioral_category"],
            "current_question_topic": question_meta["topic"],
            "current_question_difficulty": question_meta["difficulty"],
            "current_primary_competency": question_meta["primary_competency"],
            "current_expected_concepts": question_meta["expected_concepts"],
            "current_difficulty": question_meta["difficulty"],
            "current_plan_index": plan_index,
            "current_follow_up_for_plan_index": question_meta["follow_up_for_plan_index"],
            "question_metadata": state.get('question_metadata', []) + [question_meta],
            "topics_covered": topics,
            "questions": questions,
            "plan_cursor": planned_asked,
            "plan_execution_order": execution_order,
            "pending_adaptive_question": None,
            "next_difficulty_override": None,
            "remaining_question_count": remaining,
        }

    structured_llm = llm.with_structured_output(InterviewQuestion)
    
    interview_type = state['config'].get('interview_type', 'Technical')
    total_questions = state['config'].get('question_count', 5)
    question_number = len(state.get('questions', [])) + 1
    
    if interview_type.lower() == "mixed":
        expected_type = "Technical" if question_number % 2 != 0 else "Behavioral"
    elif interview_type.lower() == "technical":
        expected_type = "Technical"
    else:
        expected_type = "Behavioral"
    
    type_instructions = ""
    if expected_type == "Behavioral":
        type_instructions = "MUST ask a BEHAVIORAL question about soft skills (e.g. teamwork, conflict, leadership, failure, deadlines, ownership). DO NOT ask them to explain technical implementations. You can ask general behavioral questions; do NOT force their resume keywords into every question."
    elif expected_type == "Technical":
        type_instructions = "MUST ask a TECHNICAL question (e.g. system design, coding, architecture, debugging). You can ask about general backend concepts; do NOT feel forced to only ask about the specific technologies on their resume."

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert interviewer. Your goal is to generate the next interview question. "
                   "Question {question_number} of {total_questions}. "
                   "Expected Question Type: {expected_type}. {type_instructions} "
                   "Generate a {difficulty} question for a {role} with {experience} experience. "
                   "Candidate Context: {summary} (Technologies: {technologies}). You may use this context, but DO NOT artificially shoehorn these keywords into every question. "
                   "Previous topics covered: {topics}. "
                   "Avoid asking a substantially identical question to one already asked: {past_questions}. "
                   "Keep the question EXTREMELY short and conversational (maximum 2-3 sentences). "
                   "Ensure the question is contextual and intelligent."),
        ("human", "Generate the next question.")
    ])
    
    chain = prompt | structured_llm
    
    for attempt in range(3):
        try:
            past_questions = " | ".join(state.get('questions', [])) if state.get('questions') else "None"
            topics = ", ".join(state.get('topics_covered', [])) if state.get('topics_covered') else "None"
            technologies = ", ".join(state['profile'].get('interviewable_technologies', [])) if state['profile'].get('interviewable_technologies') else "General principles"
            summary = state['profile'].get('resume_summary') or "No summary provided."
            
            result = chain.invoke({
                "question_number": question_number,
                "total_questions": total_questions,
                "expected_type": expected_type,
                "type_instructions": type_instructions,
                "difficulty": state.get('current_difficulty', 'Medium'),
                "role": state['config']['target_role'],
                "experience": state['config']['experience_level'],
                "summary": summary,
                "technologies": technologies,
                "topics": topics,
                "past_questions": past_questions
            })
            
            # Deterministic check
            if result.question_type != expected_type:
                logger.warning(f"Generated {result.question_type} instead of {expected_type}. Retrying.")
                continue
                
            question_meta = {
                "question": result.question,
                "question_type": result.question_type,
                "behavioral_category": result.behavioral_category,
                "topic": result.topic,
                "primary_competency": result.primary_competency or result.topic,
                "difficulty": result.difficulty,
                "expected_concepts": result.expected_concepts
            }
            
            return {
                "current_question": result.question,
                "current_question_type": result.question_type,
                "current_behavioral_category": result.behavioral_category,
                "current_question_topic": result.topic,
                "current_question_difficulty": result.difficulty,
                "current_primary_competency": result.primary_competency or result.topic,
                "current_expected_concepts": result.expected_concepts,
                "question_metadata": state.get('question_metadata', []) + [question_meta],
                "topics_covered": state.get('topics_covered', []) + [result.topic],
                "questions": state.get('questions', []) + [result.question]
            }
        except Exception:
            logger.error("Question generation attempt %s failed.", attempt)
            
    # Fallback question
    if expected_type == "Technical":
        fallback_q = "How would you approach debugging a slow backend API?"
        fallback_topic = "Debugging & Performance"
    else:
        fallback_q = "Tell me about a time you faced a difficult situation at work or on a project and how you handled it."
        fallback_topic = "Problem Solving / Conflict Resolution"
        
    fallback_meta = {
        "question": fallback_q,
        "question_type": expected_type,
        "behavioral_category": "EXPERIENCE_BASED" if expected_type == "Behavioral" else None,
        "topic": fallback_topic,
        "primary_competency": fallback_topic,
        "difficulty": state.get('current_difficulty', 'Medium'),
        "expected_concepts": ["Clear communication", "Logical approach"]
    }
    
    return {
        "current_question": fallback_q,
        "current_question_type": expected_type,
        "current_behavioral_category": fallback_meta["behavioral_category"],
        "current_question_topic": fallback_topic,
        "current_question_difficulty": state.get('current_difficulty', 'Medium'),
        "current_primary_competency": fallback_topic,
        "current_expected_concepts": fallback_meta["expected_concepts"],
        "question_metadata": state.get('question_metadata', []) + [fallback_meta],
        "topics_covered": state.get('topics_covered', []) + [fallback_topic],
        "questions": state.get('questions', []) + [fallback_q]
    }


def _adaptive_context(state: GraphState, source_index):
    """Build a compact, same-competency evidence summary for the next decision."""
    questions = state.get("questions") or []
    answers = state.get("answers") or []
    analyses = state.get("analyses") or []
    metadata = state.get("question_metadata") or []
    plan_index = source_index
    entries = []
    for question_text, answer, raw_analysis, meta in zip(questions, answers, analyses, metadata):
        if not isinstance(meta, dict):
            continue
        item_plan_index = meta.get("plan_index", meta.get("source_plan_index"))
        if plan_index is not None and item_plan_index != plan_index:
            continue
        if hasattr(raw_analysis, "model_dump"):
            raw_analysis = raw_analysis.model_dump(mode="json")
        raw_analysis = raw_analysis if isinstance(raw_analysis, dict) else {}
        adaptive = raw_analysis.get("adaptive_decision") or {}
        if hasattr(adaptive, "model_dump"):
            adaptive = adaptive.model_dump(mode="json")
        adaptive = adaptive if isinstance(adaptive, dict) else {}
        missing = adaptive.get("missing_concepts", [])
        if not isinstance(missing, list):
            missing = []
        entries.append({
            "question": str(question_text)[:240],
            "answer": str(answer)[:420],
            "classification": raw_analysis.get("answer_classification", "unknown"),
            "feedback": str(raw_analysis.get("feedback", ""))[:240],
            "missing_concepts": [str(item) for item in missing],
        })

    prior_evidence = "\n".join(
        f"Q: {item['question']} | A: {item['answer']} | class: {item['classification']} | "
        f"feedback: {item['feedback']} | unresolved: {', '.join(item['missing_concepts'])}"
        for item in entries[-3:]
    ) or "No earlier scored answer for this competency."

    source_questions = [
        str(question_text) for question_text, meta in zip(questions, metadata)
        if isinstance(meta, dict)
        and (plan_index is None or meta.get("plan_index", meta.get("source_plan_index")) == plan_index)
        and str(question_text) != str(state.get("current_question") or "")
    ]
    prior_questions = " | ".join(source_questions[-5:]) or "None"

    conversations = [
        item for item in (state.get("conversation_history") or [])
        if isinstance(item, dict) and item.get("question") == state.get("current_question")
        and item.get("intent") in {"REPEAT", "CLARIFY", "HINT", "CORRECTION"}
    ][-3:]
    conversation_context = " | ".join(
        f"Candidate requested {item.get('intent', '').lower()} (interaction only; not evidence)"
        for item in conversations
    ) or "None"
    return prior_evidence, prior_questions, conversation_context

def analyze_answer_node(state: GraphState):
    answer = state.get('current_answer') or ""
    common = {
        "answers": state.get('answers', []) + [answer],
        "current_answer": None,
    }

    if not answer.strip():
        blank = AnswerAnalysis(
            correctness_score=0,
            depth_score=0,
            communication_score=0,
            feedback="No answer was provided, so there is no response evidence to assess. You can take a moment to think or say when you are unsure.",
            is_satisfactory=False,
            answer_classification="blank",
        )
        return {"analyses": state.get('analyses', []) + [blank.model_dump()], **common}

    question_type = state.get('current_question_type') or "Technical"
    if question_type not in {"Technical", "Behavioral"}:
        question_type = "Technical"
    rubric = (
        "TECHNICAL RUBRIC: assess correctness, depth, problem-solving, practical implementation, edge cases, and engineering trade-offs where relevant. Evaluate the technical answer; do not substitute a behavioral-only rubric."
        if question_type == "Technical"
        else "BEHAVIORAL RUBRIC: assess relevance, specificity, the candidate's personal contribution, reasoning, communication, collaboration, ownership, outcome, and learning as relevant. Use STAR as a flexible guide, not a required format. Evaluate the behavior; do not judge the answer as a system-design or implementation response."
    )
    prompt_messages = [
        ("system", "You are a fair, respectful interviewer evaluating one answer and choosing the next interview action.\n"
                   "Question type: {question_type}\nQuestion topic: {topic}\nPrimary competency: {primary_competency}\n"
                   "Question difficulty: {difficulty}\nTarget role: {role}\nCandidate experience level: {experience}\n"
                   "Relevant candidate context: {candidate_context}\n"
                   "Question: {question}\nExpected concepts for guidance: {expected_concepts}\n\n"
                   "{rubric}\n{uncertainty_instruction}\n"
                   "Expected concepts guide evaluation; do not demand exact keywords or a rigid checklist. Accept equivalent correct explanations and reasonable alternative approaches. Calibrate expectations to role, experience, and difficulty.\n"
                   "Assess the answer's evidence against the expected concepts, competency, question type, difficulty, and relevant candidate context. Distinguish correct and sufficiently explained, correct but shallow, partially correct, incorrect, relevant but vague, unsupported claims, contradictions, and answers that are sound but verbose. Do not reward length or penalize brevity by itself; do not penalize uncertainty when the technical content is correct. Treat a claim as unsupported only when the candidate asserts it without explaining evidence or reasoning, not merely because the answer is short. Feedback should identify what the candidate actually demonstrated and the most useful next step.\n"
                   "For the adaptive decision, use current evidence plus the compact prior evidence for this same competency. Expected concepts already adequately addressed should not be probed again. Hint requests and other conversational context are interaction only, never scored evidence. Choose ASK_FOLLOWUP only when a specific unresolved gap, unsupported claim, contradiction, or meaningful extension is worth probing; put that purpose in rationale and name the unresolved item in missing_concepts. A strong answer may merit a focused trade-off, scale, or failure-mode extension, but never ask a harder question solely because the score is high. Advance when the expected evidence is sufficient or another probe has low value.\n"
                   "Explicit requested next action: {requested_action_instruction}. Policy: for a plain ADMITTED_UNKNOWN with no previous follow-up for this source, recommend one accessible, focused ASK_FOLLOWUP at the same competency if budget remains. For an explicit EASIER request, recommend one supported lower-difficulty follow-up; if the source is already Easy, do not invent a lower level. For a weak answer, recommend a follow-up only when concrete missing concepts and insufficient competency evidence make another probe useful; a low score alone is not enough. For a strong answer, ADVANCE unless a specific unresolved gap is worth probing. Once a per-source or interview limit is reached, ADVANCE. Source follow-ups already used: {source_follow_up_count}; interview follow-ups already used: {interview_follow_up_count}. Explicit FOLLOW_UP requests remain subject to relevance, validation, and all limits.\n"
                   "A follow-up must ask one primary question, use the candidate's actual answer where helpful, target the named purpose, and remain within this question's competency, type, subtype, and assessment objective. Do not restate the original question without a clear reason, lead toward the expected answer, invent project details, or repeat a previous follow-up or resolved concept. Do not use difficulty escalation as a substitute for a purpose.\n"
                   "Previous same-competency questions: {past_questions}\nPrior scored evidence (not the current answer): {prior_evidence}\nNon-scoring conversation context: {conversation_context}\n"
                    "Use correctness_score for technical correctness on Technical questions and relevance/quality of behavioral evidence on Behavioral questions. Use depth_score for technical reasoning or behavioral specificity and reflection, as appropriate. Score communication separately. Scores are integers from 0 to 10.\n\n"
                    "Also return adaptive_decision as an object with action (ASK_FOLLOWUP, ADVANCE, or FINISH), rationale, missing_concepts (list), competency_sufficient (boolean), source_plan_index (zero-based integer), follow_up_is_duplicate (boolean), and follow_up_question (full question metadata or null). If action is ASK_FOLLOWUP, follow_up_question.question MUST be a non-empty string and the object must include the complete question metadata; if you cannot provide it, choose ADVANCE and set follow_up_question=null. Use the actual answer and analysis. Advance when sufficient evidence is present; do not probe merely because more questions are possible. Ask a focused follow-up only when it is likely to add useful evidence or resolve a specific gap. Follow-ups must use the candidate's actual response, preserve type, behavioral subtype, difficulty, topic, and primary competency, and must not repeat a previous question or resolved concept. For Behavioral questions preserve the subtype and probe only that subtype; never turn it into technical design. Never invent candidate experiences. The follow-up question should be null unless action is ASK_FOLLOWUP. A FINISH action is permitted only when no planned questions remain; otherwise choose ADVANCE. Use source_plan_index={source_plan_index}. Prior asked questions: {past_questions}. Remaining planned questions: {remaining_plan}. Prior evidence: {prior_evidence}. If there is no confirmed plan, use source_plan_index=null and choose ADVANCE unless finishing is clearly appropriate."),
        ("human", "Candidate's answer:\n{answer}")
    ]

    config = state.get('config') or {}
    concepts = state.get('current_expected_concepts') or []
    source_index = state.get('current_follow_up_for_plan_index') if state.get('current_follow_up_for_plan_index') is not None else state.get('current_plan_index')
    prior_evidence, past_questions, conversation_context = _adaptive_context(state, source_index)
    profile = state.get("profile") or {}
    if hasattr(profile, "model_dump"):
        profile = profile.model_dump(mode="json")
    summary = str(profile.get("resume_summary") or "")[:600]
    profile_skills = profile.get("skills") or []
    candidate_context = "; ".join(part for part in (
        f"resume summary: {summary}" if summary else "",
        f"relevant skills: {', '.join(str(skill) for skill in profile_skills[:12])}" if profile_skills else "",
    ) if part) or "No additional candidate context supplied."
    invocation = {
        "question_type": question_type,
        "topic": state.get('current_question_topic') or "General",
        "primary_competency": state.get('current_primary_competency') or state.get('current_question_topic') or "General",
        "difficulty": state.get('current_question_difficulty') or state.get('current_difficulty', 'Medium'),
        "role": config.get('target_role', 'Unspecified role'),
        "experience": config.get('experience_level', 'Unspecified experience level'),
        "candidate_context": candidate_context,
        "expected_concepts": ", ".join(concepts) if concepts else "No expected concepts supplied; use sound judgment.",
        "uncertainty_instruction": (
            "The candidate explicitly admitted not knowing. Acknowledge this briefly and respectfully, without shame, "
            "a lecture, generic praise, or revealing the ideal answer. Follow the bounded policy: offer one accessible "
            "same-competency follow-up when no follow-up has yet been used for this source and limits permit; do not invent a hint."
            if (state.get("current_turn_intent") or {}).get("intent") == "ADMITTED_UNKNOWN" else ""
        ),
        "requested_action_instruction": (state.get("current_turn_intent") or {}).get("requested_action") or "None",
        "source_follow_up_count": max(
            int((state.get("follow_up_counts") or {}).get(str(source_index), 0)),
            sum(bool(item.get("is_follow_up")) and item.get("follow_up_for_plan_index") == source_index
                for item in state.get("question_metadata", []) if isinstance(item, dict)),
        ),
        "interview_follow_up_count": sum(
            bool(item.get("is_follow_up")) for item in state.get("question_metadata", []) if isinstance(item, dict)
        ),
        "question": state.get('current_question') or "Question unavailable",
        "source_plan_index": source_index,
        "past_questions": past_questions,
        "conversation_context": conversation_context,
        "remaining_plan": " | ".join(
            (item.get('question', '') if isinstance(item, dict) else item.question)
            for item in (state.get('interview_plan') or [])[state.get('plan_cursor', 0):]
        ),
        "prior_evidence": prior_evidence,
        "rubric": rubric,
        "answer": answer,
    }

    failures = []
    failure_kind = "provider_error"
    for attempt in range(1, 4):
        attempt_failure_kind = "provider_error"
        try:
            prompt = ChatPromptTemplate.from_messages(prompt_messages)
            structured_llm = llm.with_structured_output(AnswerAnalysis)
            chain = prompt | structured_llm
            try:
                result = chain.invoke(invocation)
            except OutputParserException:
                attempt_failure_kind = "malformed_structured_output"
                raise
            try:
                analysis = AnswerAnalysis.model_validate(result)
            except (ValidationError, TypeError, ValueError):
                attempt_failure_kind = "malformed_structured_output"
                raise
            if analysis.status == "failed":
                attempt_failure_kind = "model_analysis_failure"
                raise ValueError("The model marked the analysis as failed.")
            if (state.get("current_turn_intent") or {}).get("intent") == "ADMITTED_UNKNOWN":
                analysis.answer_classification = "admitted_unknown"
            return {"analyses": state.get('analyses', []) + [analysis.model_dump()], **common}
        except Exception as error:
            failure_kind = attempt_failure_kind
            failures.append(failure_kind)
            logger.warning(
                "Answer analysis attempt failed; retrying: attempt=%s failure_kind=%s exception_type=%s details=%s",
                attempt, failure_kind, type(error).__name__, _analysis_error_details(error),
            )

    logger.error("Answer analysis unavailable after bounded retries: %s", "; ".join(failures))
    failed = AnswerAnalysis(
        correctness_score=0,
        depth_score=0,
        communication_score=0,
        feedback="Analysis is unavailable because the response could not be validated. No candidate score was assigned; the interview can continue without this analysis.",
        is_satisfactory=False,
        status="failed",
        adaptive_decision={"failure_kind": failure_kind},
        answer_classification="analysis_failed",
    )
    return {"analyses": state.get('analyses', []) + [failed.model_dump()], **common}

def adaptive_decision_node(state: GraphState):
    """Validate the analyzer's decision and stage a same-source follow-up.

    One structured repair is permitted for a missing, invalid, or policy-conflicting
    recommendation. Follow-ups remain linked to their source and do not consume
    or rewrite the next planned item.
    """
    plans = state.get("interview_plan", [])
    cursor = state.get("plan_cursor", 0)
    order = state.get("plan_execution_order") or list(range(len(plans)))
    analyses = state.get("analyses") or []
    metadata = state.get("question_metadata") or []
    history = list(state.get("adaptation_history", []))
    decisions = list(state.get("decisions", []))
    current_meta = metadata[-1] if metadata else {}
    source_index = state.get("current_follow_up_for_plan_index")
    if source_index is None:
        source_index = state.get("current_plan_index")
    last_analysis = analyses[-1] if analyses else {}
    turn = state.get("current_turn_intent") or {}
    intent = turn.get("intent")

    def consecutive_unknown_answers():
        count = 0
        for index in range(min(len(analyses), len(metadata)) - 1, -1, -1):
            item = metadata[index]
            if not isinstance(item, dict):
                break
            item_source = (
                item.get("follow_up_for_plan_index")
                if item.get("is_follow_up")
                else item.get("source_plan_index", item.get("plan_index"))
            )
            if item_source != source_index:
                break
            classification = analyses[index].get("answer_classification") if isinstance(analyses[index], dict) else None
            if index == len(analyses) - 1 and intent == "ADMITTED_UNKNOWN":
                classification = "admitted_unknown"
            if classification != "admitted_unknown":
                break
            count += 1
        return count

    auto_advance_unknown = (
        intent == "ADMITTED_UNKNOWN"
        and consecutive_unknown_answers() >= MAX_UNKNOWN_ANSWERS_PER_SOURCE
    )

    def decide(action, rationale, sufficient=False, missing=None, follow_up=None, duplicate=False):
        return AdaptiveDecision(
            action=action, rationale=rationale, competency_sufficient=sufficient,
            missing_concepts=missing or [], source_plan_index=source_index,
            follow_up_question=follow_up, follow_up_is_duplicate=duplicate,
        )

    def follow_up_usage():
        counts = dict(state.get("follow_up_counts", {}))
        per_source = max(
            int(counts.get(str(source_index), 0)),
            sum(bool(item.get("is_follow_up")) and item.get("follow_up_for_plan_index") == source_index
                for item in metadata if isinstance(item, dict)),
        )
        metadata_total = sum(bool(item.get("is_follow_up")) for item in metadata if isinstance(item, dict))
        counts_total = sum(max(0, int(value)) for value in counts.values())
        across_interview = max(metadata_total, counts_total)
        return per_source, across_interview

    def has_capacity():
        per_source, across_interview = follow_up_usage()
        return bool(
            plans and source_index is not None and 0 <= source_index < len(plans)
            and per_source < MAX_FOLLOW_UPS_PER_SOURCE
            and across_interview < MAX_FOLLOW_UPS_PER_INTERVIEW
        )

    def validate_decision(raw_decision):
        payload = raw_decision
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        if isinstance(payload, dict):
            payload = dict(payload)
            follow_up = payload.get("follow_up_question")
            if hasattr(follow_up, "model_dump"):
                follow_up = follow_up.model_dump(mode="json")
            if isinstance(follow_up, dict):
                follow_up = dict(follow_up)
                source = plans[source_index] if plans and source_index is not None and 0 <= source_index < len(plans) else {}
                if hasattr(source, "model_dump"):
                    source = source.model_dump(mode="json")
                active_values = {
                    "question_type": state.get("current_question_type") or current_meta.get("question_type") or source.get("question_type"),
                    "behavioral_category": state.get("current_behavioral_category") or current_meta.get("behavioral_category") or source.get("behavioral_category"),
                    "topic": state.get("current_question_topic") or current_meta.get("topic") or source.get("topic"),
                    "primary_competency": state.get("current_primary_competency") or current_meta.get("primary_competency") or source.get("primary_competency") or source.get("topic"),
                    "difficulty": state.get("current_question_difficulty") or current_meta.get("difficulty") or source.get("difficulty"),
                }
                concepts = payload.get("missing_concepts") or state.get("current_expected_concepts") or current_meta.get("expected_concepts") or source.get("expected_concepts") or []
                active_values["expected_concepts"] = concepts
                for field, value in active_values.items():
                    if (field not in follow_up or follow_up[field] is None) and value is not None:
                        follow_up[field] = value
                        if field not in metadata_defaults_applied:
                            metadata_defaults_applied.append(field)
                payload["follow_up_question"] = follow_up

        parsed = AdaptiveDecision.model_validate(payload)
        # The source index is derived from trusted graph state. Models sometimes
        # return null for this optional field even when the prompt supplies the
        # index. Bind an omitted value locally, but still reject a conflicting
        # non-null index so a follow-up can never migrate to another competency.
        if source_index is not None and parsed.source_plan_index is None:
            parsed = parsed.model_copy(update={"source_plan_index": source_index})
        if source_index is not None and parsed.source_plan_index != source_index:
            raise AdaptiveDecisionSourceMismatch(
                "Adaptive decision source did not match the current planned question"
            )
        return parsed

    metadata_defaults_applied = []

    def repair_decision(reason, missing, sufficient):
        """Make exactly one structured request for a usable recommendation."""
        source = plans[source_index] if plans and source_index is not None and 0 <= source_index < len(plans) else {}
        if hasattr(source, "model_dump"):
            source = source.model_dump(mode="json")
        per_source, across_interview = follow_up_usage()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Return one AdaptiveDecision for this answer. Respect the policy trigger and remaining limits. "
             "If ASK_FOLLOWUP, provide one novel, focused question preserving source type, behavioral subtype, topic, and competency. "
             "For EASIER, use a supported lower difficulty. Never repeat an asked question. Return ADVANCE only if no useful valid probe can be supplied. "
             "Question: {question}; source metadata: {source}; answer: {answer}; classification: {classification}; scores: {scores}; "
             "missing concepts: {missing}; competency sufficient: {sufficient}; requested action: {request}; "
             "policy trigger: {reason}; source index: {source_index}; source follow-ups used: {per_source}; "
             "interview follow-ups used: {across_interview}; asked questions: {asked_questions}."),
            ("human", "Return an AdaptiveDecision only."),
        ])
        try:
            chain = prompt | llm.with_structured_output(AdaptiveDecision)
            result = chain.invoke({
                "question": state.get("current_question") or "Question unavailable",
                "source": source,
                "answer": (state.get("answers") or [""])[-1],
                "classification": last_analysis.get("answer_classification", "unknown"),
                "scores": {
                    "correctness": last_analysis.get("correctness_score"),
                    "depth": last_analysis.get("depth_score"),
                    "communication": last_analysis.get("communication_score"),
                },
                "missing": missing,
                "sufficient": sufficient,
                "request": (state.get("current_turn_intent") or {}).get("requested_action") or "None",
                "reason": reason,
                "source_index": source_index,
                "per_source": per_source,
                "across_interview": across_interview,
                "asked_questions": state.get("questions", []),
            })
        except OutputParserException as error:
            logger.warning(
                "Adaptive repair structured parsing failed: exception_type=%s",
                type(error).__name__,
            )
            return None, "MALFORMED_REPAIR_OUTPUT", {"exception_type": type(error).__name__}
        except ValidationError as error:
            schema_errors = _safe_schema_error_context(error)
            logger.warning(
                "Adaptive repair schema validation failed during invocation: exception_type=%s fields=%s",
                type(error).__name__, schema_errors,
            )
            return None, "ADAPTIVE_SCHEMA_FAILURE", {"exception_type": type(error).__name__, "schema_errors": schema_errors}
        except APIError as error:
            status_code = getattr(error, "status_code", None)
            safe_context = {"exception_type": type(error).__name__}
            if isinstance(status_code, int):
                safe_context["http_status"] = status_code
            logger.warning(
                "Adaptive repair provider/API call failed: exception_type=%s http_status=%s",
                type(error).__name__, status_code if isinstance(status_code, int) else "unavailable",
            )
            return None, "ADAPTIVE_PROVIDER_FAILURE", safe_context
        except Exception as error:
            logger.warning(
                "Adaptive repair invocation failed outside recognized provider/parser/schema errors: exception_type=%s",
                type(error).__name__,
            )
            return None, "ADAPTIVE_INVOCATION_FAILURE", {"exception_type": type(error).__name__}

        try:
            return validate_decision(result), None, None
        except ValidationError as error:
            schema_errors = _safe_schema_error_context(error)
            logger.warning(
                "Adaptive repair result failed Pydantic validation: exception_type=%s fields=%s",
                type(error).__name__, schema_errors,
            )
            return None, "ADAPTIVE_SCHEMA_FAILURE", {"exception_type": type(error).__name__, "schema_errors": schema_errors}
        except AdaptiveDecisionSourceMismatch as error:
            logger.warning(
                "Adaptive repair result rejected by source policy: exception_type=%s",
                type(error).__name__,
            )
            return None, "ADAPTIVE_SOURCE_MISMATCH", {"exception_type": type(error).__name__}
        except Exception as error:
            logger.warning(
                "Adaptive repair result failed application validation: exception_type=%s",
                type(error).__name__,
            )
            return None, "ADAPTIVE_VALIDATION_FAILURE", {"exception_type": type(error).__name__}

    raw = last_analysis.get("adaptive_decision") if isinstance(last_analysis, dict) else None
    analysis_failed = last_analysis.get("status") == "failed" or last_analysis.get("answer_classification") == "analysis_failed"
    no_evidence = last_analysis.get("answer_classification") == "blank"
    failure_kind = None
    decision_record_context = None
    recommendation_status = "VALID"
    recommendation_origin = "MODEL"
    if auto_advance_unknown:
        decision = decide(
            "ADVANCE",
            "The candidate has twice said they do not know this competency; move to the next planned question.",
        )
        recommendation_status = "POLICY"
        recommendation_origin = "POLICY"
    elif analysis_failed:
        marker = last_analysis.get("adaptive_decision") or {}
        failure_kind = marker.get("failure_kind", "ANALYSIS_FAILURE") if isinstance(marker, dict) else "ANALYSIS_FAILURE"
        recommendation_status = "FAILURE"
        decision = decide("ADVANCE", "Answer analysis failed; no adaptive recommendation was made. Continue with the confirmed plan.")
    elif no_evidence:
        decision = decide("ADVANCE", "No answer evidence was provided; continue with the confirmed plan.")
        recommendation_status = "NOT_APPLICABLE"
        recommendation_origin = "POLICY"
    elif raw is None:
        decision = decide("ADVANCE", "The answer analysis did not include an adaptive recommendation; use the bounded repair path.")
        recommendation_status = "FAILURE"
        recommendation_origin = "MODEL"
        failure_kind = "ADAPTIVE_DECISION_MISSING"
        decision_record_context = {"reason": "answer analysis omitted adaptive_decision"}
        logger.info("Answer analysis omitted adaptive_decision; attempting bounded repair.")
    else:
        try:
            decision = validate_decision(raw)
        except ValidationError as error:
            schema_errors = _safe_schema_error_context(error)
            decision = decide("ADVANCE", "No usable adaptive recommendation was available; continue with the confirmed plan.")
            recommendation_status = "FAILURE"
            failure_kind = "ADAPTIVE_SCHEMA_FAILURE"
            decision_record_context = {"initial_exception_type": type(error).__name__, "schema_errors": schema_errors}
            missing_question_text = any(
                isinstance(item, dict) and item.get("field") == "follow_up_question.question"
                for item in schema_errors
            )
            if missing_question_text:
                logger.info("Adaptive recommendation omitted follow_up_question.question; attempting bounded repair.")
            else:
                logger.warning("Adaptive recommendation schema validation failed: exception_type=%s fields=%s", type(error).__name__, schema_errors)
        except AdaptiveDecisionSourceMismatch as error:
            decision = decide("ADVANCE", "The adaptive recommendation targeted a different planned question; continue with the confirmed plan.")
            recommendation_status = "FAILURE"
            failure_kind = "ADAPTIVE_SOURCE_MISMATCH"
            decision_record_context = {"initial_exception_type": type(error).__name__}
            logger.warning("Adaptive recommendation source validation failed: exception_type=%s", type(error).__name__)
        except Exception as error:
            decision = decide("ADVANCE", "No usable adaptive recommendation was available; continue with the confirmed plan.")
            recommendation_status = "FAILURE"
            failure_kind = "ADAPTIVE_VALIDATION_FAILURE"
            decision_record_context = {"initial_exception_type": type(error).__name__}
            logger.warning("Adaptive recommendation application validation failed: exception_type=%s", type(error).__name__)

    requested_action = turn.get("requested_action")
    per_source, _ = follow_up_usage()
    capacity = has_capacity()
    follow_up_candidate = bool(
        decision.action == "ASK_FOLLOWUP" and decision.follow_up_question is not None
        and not decision.competency_sufficient and not decision.follow_up_is_duplicate
    )
    source_for_policy = plans[source_index] if plans and source_index is not None and 0 <= source_index < len(plans) else {}
    if hasattr(source_for_policy, "model_dump"):
        source_for_policy = source_for_policy.model_dump(mode="json")
    source_difficulty_for_policy = (
        current_meta.get("difficulty") or source_for_policy.get("difficulty")
        if isinstance(current_meta, dict) else source_for_policy.get("difficulty")
    )
    proposed_difficulty = (
        decision.follow_up_question.difficulty
        if decision.follow_up_question is not None else None
    )
    easier_candidate_valid = bool(
        follow_up_candidate and source_difficulty_for_policy in {"Medium", "Hard"}
        and proposed_difficulty in {"Easy", "Medium", "Hard"}
        and {"Easy": 0, "Medium": 1, "Hard": 2}[proposed_difficulty]
        < {"Easy": 0, "Medium": 1, "Hard": 2}[source_difficulty_for_policy]
    )
    plain_unknown_needs_probe = bool(
        intent == "ADMITTED_UNKNOWN" and requested_action is None and per_source == 0
        and capacity and not follow_up_candidate
    )
    easier_needs_probe = bool(requested_action == "EASIER" and capacity and not easier_candidate_valid)
    weak_missing_gap = bool(
        min(last_analysis.get("correctness_score", 10), last_analysis.get("depth_score", 10)) <= 3
        and decision.missing_concepts and not decision.competency_sufficient and capacity
        and not follow_up_candidate
    )
    invalid_recommendation = recommendation_status == "FAILURE" and not analysis_failed and not no_evidence
    policy_reason = None if auto_advance_unknown else (
        "one bounded probe after a plain admission of not knowing" if plain_unknown_needs_probe else
        "an explicit easier-question request" if easier_needs_probe else
        "a low-evidence answer with named unresolved concepts" if weak_missing_gap else None
    )
    if not analysis_failed and not no_evidence and capacity and (invalid_recommendation or policy_reason):
        repaired, repair_error, repair_context = repair_decision(
            policy_reason or "the model recommendation was missing or invalid",
            decision.missing_concepts,
            decision.competency_sufficient,
        )
        if repaired is not None:
            decision = repaired
            recommendation_status = "VALID"
            recommendation_origin = "BOUNDED_REPAIR"
            failure_kind = None
        else:
            recommendation_status = "FAILURE"
            failure_kind = repair_error or "ADAPTIVE_REPAIR_FAILURE"
            decision = decide("ADVANCE", "The bounded adaptive repair failed; continue with the confirmed plan.")
            if repair_context:
                decision_record_context = (
                    {"initial_recommendation": decision_record_context, "repair": repair_context}
                    if decision_record_context else repair_context
                )
    elif recommendation_status == "FAILURE":
        decision = decide("ADVANCE", "The adaptive recommendation was invalid; continue with the confirmed plan.")

    if metadata_defaults_applied:
        decision_record_context = {
            **(decision_record_context or {}),
            "source_metadata_defaults_applied": sorted(metadata_defaults_applied),
        }

    # Early finish is not part of the plan-first product policy. Finish only
    # after the confirmed sequence is exhausted; otherwise advance.
    if decision.action == "FINISH" and plans and cursor < len(order):
        decision = decide("ADVANCE", "The confirmed plan still has unasked questions; continue with the next planned question.", decision.competency_sufficient, decision.missing_concepts)

    history_entry = {
        "action": decision.action,
        "rationale": decision.rationale,
        "source_plan_index": source_index,
        "missing_concepts": decision.missing_concepts,
        "competency_sufficient": decision.competency_sufficient,
        "recommendation_status": recommendation_status,
        "recommendation_origin": recommendation_origin,
        "applied": False,
    }
    if failure_kind:
        history_entry["failure_kind"] = failure_kind
    if decision_record_context:
        history_entry["failure_context"] = decision_record_context
    decision_record = decision.model_dump(mode="json")
    decision_record.update({
        "recommendation_status": recommendation_status,
        "recommendation_origin": recommendation_origin,
    })
    if failure_kind:
        decision_record["failure_kind"] = failure_kind
    if decision_record_context:
        decision_record["failure_context"] = decision_record_context
    update = {
        "decisions": decisions + [decision_record],
        "adaptation_history": history,
        "pending_adaptive_question": None,
    }

    if decision.action == "ASK_FOLLOWUP":
        per_source, existing_followups = follow_up_usage()
        counts = dict(state.get("follow_up_counts", {}))
        count = per_source
        limit_reached = count >= MAX_FOLLOW_UPS_PER_SOURCE or existing_followups >= MAX_FOLLOW_UPS_PER_INTERVIEW
        invalid_follow_up = (
            source_index is None or not plans or not 0 <= source_index < len(plans)
            or decision.competency_sufficient or decision.follow_up_question is None
        )
        if limit_reached or invalid_follow_up or decision.follow_up_is_duplicate:
            if limit_reached:
                failure_kind = "FOLLOW_UP_LIMIT_REACHED"
                rejection_context = "per-source or interview-wide follow-up limit exhausted"
                history_entry["reason"] = "The follow-up limit is exhausted; advancing."
            elif decision.follow_up_is_duplicate:
                failure_kind = "FOLLOW_UP_DUPLICATE"
                rejection_context = "model marked the proposed follow-up as duplicate"
                history_entry["reason"] = "The proposed follow-up duplicates an asked question; advancing."
            else:
                failure_kind = "FOLLOW_UP_VALIDATION_FAILURE"
                rejection_context = "missing source, sufficient competency, or missing follow-up question"
                history_entry["reason"] = "The proposed follow-up was incomplete or invalid; advancing."
            history_entry["recommendation_status"] = "REJECTED"
            history_entry["failure_kind"] = failure_kind
            history_entry["failure_context"] = rejection_context
            history.append(history_entry)
            update["adaptation_history"] = history
            fallback_record = decide("ADVANCE", history_entry["reason"], decision.competency_sufficient, decision.missing_concepts).model_dump(mode="json")
            fallback_record.update({
                "recommendation_status": "REJECTED",
                "failure_kind": failure_kind,
                "failure_context": rejection_context,
            })
            update["decisions"][-1] = fallback_record
            return update

        source = plans[source_index]
        if hasattr(source, "model_dump"):
            source = source.model_dump()
        # Preserve the metadata of the question actually asked, including any
        # accepted difficulty override, rather than reverting to the plan's
        # original difficulty on a follow-up.
        if current_meta.get("plan_index") == source_index:
            source = {**source, **current_meta}
        try:
            follow_up = decision.follow_up_question
            if not follow_up.question.strip() or not follow_up.expected_concepts:
                raise ValueError("Follow-up question or expected concepts are empty")
            if follow_up.question_type != source.get("question_type"):
                raise ValueError("Follow-up changed the source interview category")
            if follow_up.behavioral_category != source.get("behavioral_category"):
                raise ValueError("Follow-up changed the source behavioral subtype")
            source_competency = (source.get("primary_competency") or source.get("topic") or "").strip().casefold()
            returned_competency = (follow_up.primary_competency or follow_up.topic or "").strip().casefold()
            if source_competency and returned_competency != source_competency:
                raise ValueError("Follow-up changed the source competency")
            normalized_question = " ".join(follow_up.question.casefold().split())
            prior_questions = {" ".join(str(item).casefold().split()) for item in state.get("questions", [])}
            if normalized_question in prior_questions:
                raise ValueError("Follow-up duplicates a previously asked question")
            # Preserve source metadata regardless of any drift in model fields.
            follow_up.topic = source.get("topic") or follow_up.topic
            follow_up.primary_competency = source.get("primary_competency") or source.get("topic")
            requested_action = (state.get("current_turn_intent") or {}).get("requested_action")
            source_difficulty = source.get("difficulty")
            if requested_action == "EASIER":
                difficulty_order = {"Easy": 0, "Medium": 1, "Hard": 2}
                proposed_difficulty = follow_up.difficulty
                if (
                    source_difficulty not in difficulty_order
                    or proposed_difficulty not in difficulty_order
                    or difficulty_order[proposed_difficulty] >= difficulty_order[source_difficulty]
                ):
                    raise ValueError("An easier request requires a supported lower-difficulty follow-up")
            else:
                # Ignore model-proposed difficulty drift unless the candidate
                # explicitly requested an easier follow-up.
                follow_up.difficulty = source_difficulty
        except Exception as error:
            duplicate = "duplicate" in str(error).casefold()
            failure_kind = "FOLLOW_UP_DUPLICATE" if duplicate else "FOLLOW_UP_VALIDATION_FAILURE"
            rejection_context = "duplicate question" if duplicate else "question content or source metadata validation failed"
            logger.info(
                "Adaptive follow-up rejected; advancing: failure_kind=%s context=%s",
                failure_kind, rejection_context,
            )
            history_entry["reason"] = (
                "The proposed follow-up duplicates an asked question; advancing."
                if duplicate else
                "The proposed follow-up did not preserve source metadata; advancing."
            )
            history_entry["recommendation_status"] = "REJECTED"
            history_entry["failure_kind"] = failure_kind
            history_entry["failure_context"] = rejection_context
            history.append(history_entry)
            update["adaptation_history"] = history
            fallback_record = decide("ADVANCE", history_entry["reason"], decision.competency_sufficient, decision.missing_concepts).model_dump(mode="json")
            fallback_record.update({
                "recommendation_status": "REJECTED",
                "failure_kind": failure_kind,
                "failure_context": rejection_context,
            })
            update["decisions"][-1] = fallback_record
            return update

        counts[str(source_index)] = count + 1
        update["follow_up_counts"] = counts
        if requested_action == "EASIER":
            # Question generation consumes this existing one-turn override;
            # it is cleared as soon as the follow-up becomes active.
            update["next_difficulty_override"] = follow_up.difficulty
        update["pending_adaptive_question"] = {
            "question": follow_up.model_dump(mode="json"),
            "is_follow_up": True,
            "source_plan_index": source_index,
            "rationale": decision.rationale,
        }
        history_entry.update({"applied": True, "follow_up_count": count + 1, "follow_up_question": follow_up.question})

    elif decision.action == "FINISH":
        history_entry["applied"] = True
    else:
        history_entry["applied"] = True

    history.append(history_entry)
    update["adaptation_history"] = history
    return update

class StateConsistencyError(ValueError):
    """Raised when evaluation history cannot be aligned without guessing."""


def validate_evaluation_state(state: GraphState):
    """Validate parallel interview histories and return typed usable entries."""
    questions = state.get("questions", [])
    answers = state.get("answers", [])
    raw_analyses = state.get("analyses", [])
    metadata = state.get("question_metadata", [])
    plan = state.get("interview_plan", [])
    config = state.get("config", {})
    expected = config.get("question_count")

    if not all(isinstance(items, list) for items in (questions, answers, raw_analyses, metadata, plan)):
        raise StateConsistencyError("Interview history fields must be lists.")
    counts = {
        "questions": len(questions), "answers": len(answers),
        "analyses": len(raw_analyses), "question metadata": len(metadata),
    }
    if len(set(counts.values())) != 1:
        detail = ", ".join(f"{name}={count}" for name, count in counts.items())
        raise StateConsistencyError(f"Interview history lengths do not align: {detail}.")
    follow_up_count = sum(bool(item.get("is_follow_up")) for item in metadata if isinstance(item, dict))
    planned_asked_count = len(questions) - follow_up_count
    if expected is not None and planned_asked_count > expected:
        raise StateConsistencyError(f"Asked {planned_asked_count} planned questions, exceeding configured count {expected}.")
    if follow_up_count > MAX_FOLLOW_UPS_PER_INTERVIEW:
        raise StateConsistencyError("Interview history exceeds the hidden follow-up safety ceiling.")
    if plan and expected is not None and len(plan) != expected:
        raise StateConsistencyError(f"Confirmed plan has {len(plan)} questions; configuration expects {expected}.")
    if plan and state.get("plan_cursor", planned_asked_count) != planned_asked_count:
        raise StateConsistencyError("Plan cursor does not match the number of asked planned questions.")
    if plan:
        execution_order = state.get("plan_execution_order") or list(range(len(plan)))
        if len(execution_order) != len(plan) or set(execution_order) != set(range(len(plan))):
            raise StateConsistencyError("Plan execution order must contain every planned question exactly once.")

    typed_analyses = []
    selected_type = config.get("interview_type", "Mixed")
    for index, (question, answer, raw_analysis, meta) in enumerate(zip(questions, answers, raw_analyses, metadata), start=1):
        if not isinstance(meta, dict):
            raise StateConsistencyError(f"Question metadata item {index} is not an object.")
        if meta.get("question") != question:
            raise StateConsistencyError(f"Question metadata item {index} does not match the asked question.")
        category = meta.get("question_type")
        if category not in {"Technical", "Behavioral"}:
            raise StateConsistencyError(f"Question metadata item {index} has an invalid category.")
        if selected_type in {"Technical", "Behavioral"} and category != selected_type:
            raise StateConsistencyError(f"Question {index} category {category} conflicts with selected interview type {selected_type}.")
        if plan:
            plan_index = meta.get("plan_index", index - 1)
            if not isinstance(plan_index, int) or not 0 <= plan_index < len(plan):
                raise StateConsistencyError(f"Question metadata item {index} has an invalid plan index.")
            planned = plan[plan_index]
            if hasattr(planned, "model_dump"):
                planned = planned.model_dump()
            if not meta.get("is_follow_up") and planned.get("question") != question:
                raise StateConsistencyError(f"Asked question {index} differs from its confirmed plan item without a follow-up record.")
            if meta.get("is_follow_up"):
                source_index = meta.get("follow_up_for_plan_index")
                if not isinstance(source_index, int) or not 0 <= source_index < len(plan):
                    raise StateConsistencyError(f"Follow-up question {index} is missing its source plan index.")
                if meta.get("plan_index") != source_index or meta.get("source_plan_index", source_index) != source_index:
                    raise StateConsistencyError(f"Follow-up question {index} has inconsistent source metadata.")
                source = plan[source_index]
                if hasattr(source, "model_dump"):
                    source = source.model_dump()
                if source.get("question_type") != category:
                    raise StateConsistencyError(f"Follow-up question {index} changed category from its source question.")
                if meta.get("behavioral_category") != source.get("behavioral_category"):
                    raise StateConsistencyError(f"Follow-up question {index} changed behavioral subtype from its source question.")
                if (meta.get("primary_competency") or "").strip().casefold() != (source.get("primary_competency") or source.get("topic") or "").strip().casefold():
                    raise StateConsistencyError(f"Follow-up question {index} changed competency from its source question.")
        try:
            typed_analyses.append(AnswerAnalysis.model_validate(raw_analysis))
        except Exception as error:
            raise StateConsistencyError(f"Answer analysis {index} is invalid: {error}") from error
    return list(zip(questions, answers, typed_analyses, metadata))


def _confidence_from_evidence(usable, competencies):
    count = len(usable)
    breadth = len(competencies)
    if count < 3:
        return "Very Low"
    if count < 4 or breadth < 2:
        return "Low"
    if count < 6 or breadth < 3:
        return "Moderate"
    return "High"


def _mean(values):
    return round(sum(values) / len(values), 2) if values else None


def evaluate_node(state: GraphState):
    """Build a deterministic, evidence-grounded report from aligned histories.

    Per-answer evidence score (0-10) = 0.50 * correctness/relevance +
    0.30 * depth + 0.20 * communication. Overall score is the mean of usable
    answer evidence scores multiplied by 10, yielding a bounded 0-100 scale.
    """
    try:
        aligned = validate_evaluation_state(state)
    except StateConsistencyError as error:
        logger.error("Evaluation state consistency check failed: %s", error)
        return {"status": "COMPLETED_WITH_ERRORS", "evaluation_error": str(error)}

    usable = []
    insufficient = []
    partial = []
    demonstrated = []
    knowledge_gaps = []
    config = state.get("config", {})
    for question_index, (question, answer, analysis, meta) in enumerate(aligned, start=1):
        competency = meta.get("primary_competency") or meta.get("topic") or "Assessed competency"
        label = f"{competency} (question {question_index})"
        if analysis.status == "failed" or analysis.answer_classification == "analysis_failed":
            insufficient.append(f"{label}: answer analysis failed; no candidate score was assigned.")
            continue
        if analysis.answer_classification == "blank":
            insufficient.append(f"{label}: no answer was provided, so there was no candidate evidence to score.")
            continue
        if analysis.answer_classification == "irrelevant":
            insufficient.append(f"{label}: the response did not address the assessed competency, so no candidate score was assigned.")
            continue
        usable.append((question, answer, analysis, meta, label))
        category = meta["question_type"]
        if category == "Technical" and (analysis.answer_classification == "admitted_unknown" or analysis.correctness_score <= 3):
            knowledge_gaps.append(f"{label}: the answer showed a gap in the assessed technical area.")
        elif analysis.correctness_score >= 7 and analysis.depth_score >= 6:
            demonstrated.append(f"{label}: the answer provided relevant, sufficiently detailed evidence.")
        elif analysis.answer_classification == "admitted_unknown" and category == "Behavioral":
            insufficient.append(f"{label}: the candidate did not provide behavioral evidence for this question.")
        elif analysis.correctness_score <= 3 and category == "Behavioral":
            insufficient.append(f"{label}: the answer did not provide enough relevant behavioral evidence.")
        else:
            partial.append(f"{label}: the answer provided some relevant evidence but did not fully establish the competency.")

    competencies = {
        str(meta.get("primary_competency") or meta.get("topic") or "").strip().casefold()
        for _, _, _, meta, _ in usable
        if (meta.get("primary_competency") or meta.get("topic"))
    }
    confidence = _confidence_from_evidence(usable, competencies)

    def evidence_score(analysis):
        return 0.50 * analysis.correctness_score + 0.30 * analysis.depth_score + 0.20 * analysis.communication_score

    technical = [entry[2] for entry in usable if entry[3]["question_type"] == "Technical"]
    behavioral = [entry[2] for entry in usable if entry[3]["question_type"] == "Behavioral"]
    communication_score = _mean([item[2].communication_score for item in usable])
    technical_score = _mean([item.correctness_score for item in technical])
    problem_solving_score = _mean([0.60 * item.correctness_score + 0.40 * item.depth_score for item in technical])
    behavioral_score = _mean([evidence_score(item) for item in behavioral])
    overall_score = _mean([evidence_score(item) * 10 for _, _, item, _, _ in usable])

    if len(usable) < 3:
        readiness = "Insufficient evidence"
        readiness_basis = f"Only {len(usable)} responses supplied usable evidence; one or two questions cannot establish overall role readiness."
    elif knowledge_gaps:
        readiness = "Gaps observed in assessed areas"
        readiness_basis = "One or more technical knowledge gaps were demonstrated in the assessed questions; this is not a judgment of general suitability."
    elif confidence in {"Very Low", "Low"}:
        readiness = "Insufficient evidence"
        readiness_basis = f"Evidence breadth is limited ({len(competencies)} distinct competencies); overall role readiness is not established."
    elif overall_score is not None and overall_score >= 75:
        readiness = "Positive evidence in assessed areas"
        readiness_basis = "Responses provided positive evidence across the assessed areas; this is not a complete measure of overall role readiness."
    else:
        readiness = "Mixed evidence in assessed areas"
        readiness_basis = "The available answers show mixed evidence; conclusions apply only to the competencies assessed."

    if not usable:
        failed_count = sum(
            item[2].status == "failed" or item[2].answer_classification == "analysis_failed"
            for item in aligned
        )
        insufficient_answer_count = sum(
            item[2].status != "failed" and item[2].answer_classification in {"blank", "irrelevant"}
            for item in aligned
        )
        reasons = []
        if failed_count:
            reasons.append(f"analysis failed for {failed_count} answer(s)")
        if insufficient_answer_count:
            reasons.append(f"{insufficient_answer_count} response(s) were blank or irrelevant")
        reason_text = "; ".join(reasons) or "no response supplied usable evidence"
        final_feedback = (
            f"No usable answer evidence was available: {reason_text}. "
            "No candidate score or overall readiness judgment was assigned."
        )
    else:
        final_feedback = (
            f"Usable evidence was available for {len(usable)} of {len(aligned)} answers across {len(competencies)} distinct competencies. "
            f"Confidence is {confidence.lower()}. Findings describe only the assessed answers and do not establish broader personal traits or general suitability."
        )

    profile = state.get("profile", {})
    report = EvaluationReport(
        overall_score=overall_score,
        technical_score=technical_score,
        problem_solving_score=problem_solving_score,
        communication_score=communication_score,
        behavioral_score=behavioral_score,
        evaluation_confidence=confidence,
        resume_claims=profile.get("inferred_strengths", []),
        interview_demonstrated_strengths=demonstrated,
        weaknesses=partial + knowledge_gaps + insufficient,
        technical_gaps=knowledge_gaps,
        recommended_topics=[entry.split(":", 1)[0] for entry in knowledge_gaps],
        role_readiness=readiness,
        final_feedback=final_feedback,
        evidence_summary={
            "demonstrated_competence": demonstrated,
            "partial_evidence": partial,
            "insufficient_evidence": insufficient,
            "demonstrated_knowledge_gaps": knowledge_gaps,
        },
        readiness_basis=readiness_basis,
    )
    return {"evaluation": report.model_dump(mode="json"), "status": "COMPLETED", "evaluation_error": None}

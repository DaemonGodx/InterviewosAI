from unittest.mock import MagicMock

import pytest

from app.engine.graph import InterviewEngine
from app.engine import nodes
from app.models.schemas import InterviewQuestion, TurnIntent


def planned_question(text):
    return InterviewQuestion(
        question=text, topic="Debugging", primary_competency="Incident diagnosis",
        difficulty="Medium", question_type="Technical", expected_concepts=["Logs", "Verification"],
    ).model_dump(mode="json")


def interview_state():
    plan = [planned_question("How would you debug an API timeout?"), planned_question("How do you verify a fix?")]
    first = plan[0]
    return {
        "session_id": "conversation-test",
        "config": {"target_role": "Backend", "experience_level": "Mid-level", "question_count": 2, "interview_type": "Technical"},
        "profile": {}, "interview_plan": plan,
        "questions": [first["question"]], "answers": [], "analyses": [], "decisions": [],
        "topics_covered": [first["topic"]], "question_metadata": [{
            "question": first["question"], "question_type": "Technical", "topic": first["topic"],
            "primary_competency": first["primary_competency"], "difficulty": "Medium",
            "expected_concepts": first["expected_concepts"], "plan_index": 0, "is_follow_up": False,
        }],
        "current_question": first["question"], "current_question_type": "Technical",
        "current_question_topic": first["topic"], "current_primary_competency": first["primary_competency"],
        "current_question_difficulty": "Medium", "current_expected_concepts": first["expected_concepts"],
        "current_plan_index": 0, "current_follow_up_for_plan_index": None,
        "plan_cursor": 1, "plan_execution_order": [0, 1], "remaining_question_count": 1,
        "follow_up_counts": {"0": 1}, "status": "IN_PROGRESS",
    }


def patch_intent(mocker, intent):
    def classify(state):
        return {
            "current_turn_intent": {"intent": intent, "confidence": 1.0},
            "current_input": None,
            "classified_utterance": state["current_input"],
            "consecutive_ambiguous_turns": (state.get("consecutive_ambiguous_turns", 0) + 1 if intent == "AMBIGUOUS" else 0),
        }
    mocker.patch("app.engine.graph.classify_turn_node", side_effect=classify)


@pytest.mark.parametrize("utterance,intent", [
    ("Could you repeat that?", "REPEAT"),
    ("What do you mean by timeout?", "CLARIFY"),
    ("Can I have a hint?", "HINT"),
    ("What is the weather?", "OFF_TOPIC"),
    ("Wait, let me correct that", "CORRECTION"),
    ("um...", "AMBIGUOUS"),
])
def test_conversational_intents_preserve_active_question_and_progress(mocker, utterance, intent):
    patch_intent(mocker, intent)
    before = interview_state()

    result = InterviewEngine().process_answer(before, utterance)

    assert result["current_turn_intent"]["intent"] == intent
    assert result["current_question"] == before["current_question"]
    assert result["question_metadata"] == before["question_metadata"]
    assert result["questions"] == before["questions"]
    assert result["answers"] == result["analyses"] == []
    assert result["plan_cursor"] == before["plan_cursor"]
    assert result["follow_up_counts"] == before["follow_up_counts"]
    assert result["status"] == "IN_PROGRESS"
    assert len(result["conversation_history"]) == 1
    assert result["interviewer_response"]
    if intent == "REPEAT":
        assert result["interviewer_response"] == before["current_question"]


def test_skip_uses_blank_analysis_and_advances_without_candidate_points(mocker):
    patch_intent(mocker, "SKIP")
    state = interview_state()

    result = InterviewEngine().process_answer(state, "skip")

    assert result["answers"] == [""]
    assert len(result["analyses"]) == len(result["answers"]) == 1
    assert len(result["question_metadata"]) == len(result["questions"]) == 2
    assert result["analyses"][0]["answer_classification"] == "blank"
    assert result["analyses"][0]["correctness_score"] == 0
    assert result["questions"][0] == state["current_question"]
    assert result["plan_cursor"] == 2
    assert result["conversation_history"][0]["intent"] == "SKIP"


def test_classification_failure_and_low_confidence_become_ambiguous(mocker):
    for result in (TurnIntent(intent="ANSWER", confidence=0.2), RuntimeError("provider unavailable")):
        chain = MagicMock()
        if isinstance(result, Exception):
            chain.invoke.side_effect = result
        else:
            chain.invoke.return_value = result
        prompt = MagicMock()
        prompt.__or__ = lambda _self, _other: chain
        mocker.patch("app.engine.nodes.ChatPromptTemplate.from_messages", return_value=prompt)
        model = MagicMock()
        model.with_structured_output.return_value = MagicMock()
        mocker.patch.object(nodes, "llm", model)
        prompt.__or__ = lambda _self, _other: chain
        state = {"current_input": "unclear", "current_question": "What happened?", "consecutive_ambiguous_turns": 1}

        update = nodes.classify_turn_node(state)

        assert update["current_turn_intent"]["intent"] == "AMBIGUOUS"
        assert update["consecutive_ambiguous_turns"] == 2


def test_go_easy_is_a_direct_easier_question_request():
    update = nodes.classify_turn_node({
        "current_input": "go easy", "current_question": "How would you design this API?",
        "current_question_difficulty": "Medium",
    })

    assert update["current_turn_intent"]["intent"] == "ADMITTED_UNKNOWN"
    assert update["current_turn_intent"]["requested_action"] == "EASIER"
    assert update["current_turn_intent"]["confidence"] == 1.0


def test_ambiguous_turns_reach_bounded_fallback():
    response = nodes.respond_to_conversation_node({
        "current_turn_intent": {"intent": "AMBIGUOUS", "confidence": 0.1},
        "classified_utterance": "unclear", "current_question": "What happened?",
        "consecutive_ambiguous_turns": nodes.MAX_AMBIGUOUS_TURNS,
    })
    assert "answer" in response["interviewer_response"].lower()
    assert "hint" in response["interviewer_response"].lower()
    assert "skip" in response["interviewer_response"].lower()


def test_second_consecutive_ambiguous_turn_auto_skips_without_scoring(mocker):
    patch_intent(mocker, "AMBIGUOUS")
    state = interview_state()
    state["consecutive_ambiguous_turns"] = 1

    result = InterviewEngine().process_answer(state, "good")

    assert result["current_turn_intent"]["intent"] == "SKIP"
    assert result["conversation_history"][-1]["automatic"] is True
    assert result["analyses"][-1]["answer_classification"] == "blank"
    assert result["answers"][-1] == ""
    assert result["current_question"] == state["interview_plan"][1]["question"]
    assert result["plan_cursor"] == 2
    assert result["consecutive_ambiguous_turns"] == 0


@pytest.mark.parametrize("intent", ["CLARIFY", "HINT"])
def test_clarification_and_hint_responses_use_question_context(mocker, intent):
    chain = MagicMock()
    chain.invoke.return_value = "Think about the observations that would narrow the cause."
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _other: chain
    prompt_builder = MagicMock(return_value=prompt)
    mocker.patch("app.engine.nodes.ChatPromptTemplate.from_messages", prompt_builder)
    mocker.patch.object(nodes, "llm", MagicMock())
    state = {
        "current_turn_intent": {"intent": intent}, "classified_utterance": "Can you help?",
        "current_question": "How would you debug an API timeout?", "current_question_topic": "Debugging",
        "current_expected_concepts": ["Logs", "Verification"],
    }

    result = nodes.respond_to_conversation_node(state)

    invocation = chain.invoke.call_args.args[0]
    assert invocation["question"] == state["current_question"]
    assert invocation["topic"] == "Debugging"
    assert result["interviewer_response"] == chain.invoke.return_value
    assert result["conversation_history"][0]["intent"] == intent
    if intent == "CLARIFY":
        assert "preserve the assessed concept" in invocation["instruction"].lower()
    else:
        assert "solution" in invocation["instruction"].lower()


def test_hints_progress_once_then_stop_before_revealing_solution(mocker):
    question = "How would you debug an API timeout?"
    history = [{
        "question": question, "utterance": "hint please", "intent": "HINT", "response": "Look at the evidence first."
    }]
    chain = MagicMock()
    chain.invoke.return_value = "Compare where the delay appears across the request path."
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _other: chain
    prompt_builder = MagicMock(return_value=prompt)
    mocker.patch("app.engine.nodes.ChatPromptTemplate.from_messages", prompt_builder)
    mocker.patch.object(nodes, "llm", MagicMock())
    state = {
        "current_turn_intent": {"intent": "HINT"}, "classified_utterance": "another hint",
        "current_question": question, "current_question_topic": "Debugging",
        "current_expected_concepts": ["Logs", "Verification"], "conversation_history": history,
    }

    second_hint = nodes.respond_to_conversation_node(state)

    instruction = chain.invoke.call_args.args[0]["instruction"]
    assert "second hint" in instruction.lower()
    assert "full solution" in instruction.lower()
    assert "Logs, Verification" == chain.invoke.call_args.args[0]["concepts"]
    assert second_hint["conversation_history"][-1]["question"] == question

    state["conversation_history"] = history + [second_hint["conversation_history"][-1]]
    chain.invoke.reset_mock()
    third_request = nodes.respond_to_conversation_node(state)
    assert not chain.invoke.called
    assert "try your approach" in third_request["interviewer_response"].lower()
    assert "skip" in third_request["interviewer_response"].lower()


@pytest.mark.parametrize("intent", ["CLARIFY", "HINT"])
def test_response_generation_failure_uses_safe_intent_specific_fallback(mocker, intent):
    chain = MagicMock()
    chain.invoke.side_effect = RuntimeError("model unavailable")
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _other: chain
    mocker.patch("app.engine.nodes.ChatPromptTemplate.from_messages", return_value=prompt)
    mocker.patch.object(nodes, "llm", MagicMock())
    response = nodes.respond_to_conversation_node({
        "current_turn_intent": {"intent": intent}, "classified_utterance": "help",
        "current_question": "How would you debug an API timeout?", "current_question_topic": "Debugging",
        "current_expected_concepts": ["Logs"],
    })["interviewer_response"]
    assert response
    assert "ideal answer" not in response.lower()
    if intent == "HINT":
        assert "concept" in response.lower()
    else:
        assert "Debugging" in response


def test_uncertain_answer_is_not_forced_into_admission_intent(mocker):
    chain = MagicMock()
    chain.invoke.return_value = TurnIntent(intent="ANSWER", confidence=0.94)
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _other: chain
    prompt_builder = MagicMock(return_value=prompt)
    mocker.patch("app.engine.nodes.ChatPromptTemplate.from_messages", prompt_builder)
    mocker.patch.object(nodes, "llm", MagicMock())
    state = {
        "current_input": "I'm not completely sure, but I would compare the request traces and database timings.",
        "current_question": "How would you debug an API timeout?",
    }
    update = nodes.classify_turn_node(state)
    assert update["current_turn_intent"]["intent"] == "ANSWER"
    assert "remains ANSWER" in prompt_builder.call_args.args[0][0][1]


@pytest.mark.parametrize("intent,expected", [
    ("ADMITTED_UNKNOWN", "explicitly admitted not knowing"),
    ("ANSWER", ""),
])
def test_answer_feedback_gets_uncertainty_guidance_only_for_explicit_admission(mocker, intent, expected):
    chain = MagicMock()
    chain.invoke.return_value = {
        "correctness_score": 5, "depth_score": 5, "communication_score": 6,
        "feedback": "The response is limited.", "is_satisfactory": False,
    }
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _other: chain
    prompt_builder = MagicMock(return_value=prompt)
    mocker.patch("app.engine.nodes.ChatPromptTemplate.from_messages", prompt_builder)
    mocker.patch.object(nodes, "llm", MagicMock())
    nodes.analyze_answer_node({
        "current_question": "How would you debug an API timeout?", "current_answer": "I don't know",
        "current_turn_intent": {"intent": intent}, "answers": [], "analyses": [],
        "current_question_type": "Technical", "current_expected_concepts": ["Logs"],
    })
    guidance = chain.invoke.call_args.args[0]["uncertainty_instruction"]
    assert expected in guidance


def test_admitted_unknown_classification_is_preserved_after_analysis(mocker):
    patch_intent(mocker, "ADMITTED_UNKNOWN")
    chain = MagicMock()
    chain.invoke.return_value = {
        "correctness_score": 0, "depth_score": 0, "communication_score": 5,
        "feedback": "Thank you for being candid.", "is_satisfactory": False,
        "answer_classification": "substantive",
    }
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _other: chain
    mocker.patch("app.engine.nodes.ChatPromptTemplate.from_messages", return_value=prompt)
    model = MagicMock()
    model.with_structured_output.return_value = MagicMock()
    mocker.patch.object(nodes, "llm", model)
    state = interview_state()
    result = InterviewEngine().process_answer(state, "I don't know")
    assert result["analyses"][0]["answer_classification"] == "admitted_unknown"


@pytest.mark.parametrize("mode", ["Text", "Voice"])
def test_cli_handles_conversation_without_requiring_analysis(mocker, mode):
    from app import __main__ as cli

    qtext = "How would you debug an API timeout?"
    state = {
        "config": {"interview_mode": mode, "question_count": 1},
        "questions": [qtext], "question_metadata": [{"is_follow_up": False}],
        "current_question": qtext, "current_difficulty": "Medium", "topics_covered": ["Debugging"],
        "remaining_question_count": 0, "status": "IN_PROGRESS", "analyses": [],
    }

    class FakeEngine:
        def start_interview(self, _initial):
            return dict(state)

        def process_answer(self, current, answer):
            if answer in {"repeat", "voice-input-1"}:
                return {**current, "current_turn_intent": {"intent": "REPEAT"},
                        "interviewer_response": qtext, "conversation_history": [{"intent": "REPEAT"}]}
            return {**current, "current_turn_intent": {"intent": "ANSWER"},
                    "analyses": [{"status": "success", "answer_classification": "substantive",
                                  "correctness_score": 7, "feedback": "Clear."}],
                    "status": "COMPLETED", "evaluation": {
                        "overall_score": 70, "technical_score": 7, "problem_solving_score": 7,
                        "communication_score": 7, "behavioral_score": None, "evaluation_confidence": "Low",
                        "resume_claims": [], "interview_demonstrated_strengths": [], "weaknesses": [],
                        "technical_gaps": [], "recommended_topics": [], "role_readiness": "Developing",
                        "final_feedback": "Continue practicing.",
                    }}

    class FakeDB:
        def save_session(self, _state):
            pass

    mocker.patch.object(cli, "InterviewEngine", FakeEngine)
    if mode == "Text":
        mocker.patch.object(cli.Prompt, "ask", side_effect=["repeat", "answer"])
    else:
        mocker.patch.object(cli, "AudioService", return_value=MagicMock())
        speech = MagicMock()
        mocker.patch.object(cli, "SpeechService", return_value=speech)
        mocker.patch.object(cli, "get_voice_answer", side_effect=["voice-input-1", "answer"])

    cli.run_interview_loop(FakeDB(), {"config": {"interview_mode": mode, "question_count": 1}})

    if mode == "Voice":
        # One synthesis for the original question and one for the repeat reply.
        assert speech.speak.call_count == 2

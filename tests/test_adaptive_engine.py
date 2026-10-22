import pytest
from unittest.mock import MagicMock
from pydantic import ValidationError

from app.engine import nodes
from app.engine.graph import InterviewEngine, route_after_decision
from app.models.schemas import AdaptiveDecision, AnswerAnalysis, InterviewQuestion


def question(text, kind="Technical", topic="Debugging", difficulty="Medium", subtype=None, competency=None):
    return InterviewQuestion(
        question=text, topic=topic, primary_competency=competency or topic,
        difficulty=difficulty, question_type=kind, behavioral_category=subtype,
        expected_concepts=["Evidence", "Reasoning"],
    ).model_dump(mode="json")


def state():
    plan = [
        question("How would you debug an API outage?", topic="Incident response", competency="Incident diagnosis"),
        question("Describe a time you resolved a team disagreement.", "Behavioral", "Teamwork", subtype="EXPERIENCE_BASED", competency="Conflict resolution"),
        question("How would you improve cache consistency?", topic="Caching"),
    ]
    return {
        "session_id": "adaptive-test",
        "config": {"target_role": "Backend Engineer", "experience_level": "Mid-level", "interview_type": "Mixed", "question_count": 3},
        "profile": {}, "interview_plan": plan, "plan_cursor": 1, "plan_execution_order": [0, 1, 2],
        "questions": [plan[0]["question"]],
        "question_metadata": [{**plan[0], "plan_index": 0, "source_plan_index": 0, "is_follow_up": False}],
        "answers": ["I checked logs and rolled back the deployment."],
        "analyses": [analysis(decision("ADVANCE", 0, sufficient=True))],
        "decisions": [], "topics_covered": ["Incident response"],
        "current_difficulty": "Medium", "current_question": plan[0]["question"],
        "current_question_type": "Technical", "current_behavioral_category": None,
        "current_question_topic": "Incident response", "current_question_difficulty": "Medium",
        "current_primary_competency": "Incident diagnosis", "current_expected_concepts": ["Evidence", "Reasoning"],
        "current_plan_index": 0, "current_follow_up_for_plan_index": None, "current_answer": None,
        "pending_adaptive_question": None, "next_difficulty_override": None,
        "follow_up_counts": {}, "adaptation_history": [], "remaining_question_count": 2,
        "status": "IN_PROGRESS",
    }


def decision(action, source, *, question_data=None, rationale="Evidence-based next step", sufficient=False, missing=None, duplicate=False):
    return {
        "action": action, "rationale": rationale, "missing_concepts": missing or [],
        "competency_sufficient": sufficient, "follow_up_question": question_data,
        "source_plan_index": source, "follow_up_is_duplicate": duplicate,
    }


def analysis(adaptive, **overrides):
    return {
        "correctness_score": 6, "depth_score": 5, "communication_score": 7,
        "feedback": "The answer has some evidence; explain the verification step.",
        "is_satisfactory": False, "status": "success", "answer_classification": "partially_correct",
        "adaptive_decision": adaptive, **overrides,
    }


def test_action_schema_is_restricted_and_validates_structured_recommendations():
    validated = AdaptiveDecision.model_validate(decision("ADVANCE", 0, sufficient=True))
    assert validated.action == "ADVANCE"
    with pytest.raises(ValidationError):
        AdaptiveDecision.model_validate({"action": "FOLLOW_UP", "reasoning": "old action", "topic": None})


def test_strong_answer_with_sufficient_evidence_advances_without_extra_call():
    interview = state()
    interview["analyses"][-1] = analysis(decision("ADVANCE", 0, sufficient=True))
    update = nodes.adaptive_decision_node(interview)
    generated = nodes.generate_question_node({**interview, **update})
    assert update["decisions"][-1]["action"] == "ADVANCE"
    assert generated["current_question"] == interview["interview_plan"][1]["question"]
    assert generated["plan_cursor"] == 2


def test_weak_answer_with_named_missing_concepts_gets_one_bounded_repair(mocker):
    interview = state()
    interview["analyses"][-1] = analysis(
        decision("ADVANCE", 0, sufficient=False, missing=["Transaction isolation"]),
        correctness_score=2, depth_score=2, communication_score=6,
    )
    follow_up = question(
        "How would you prevent two simultaneous bookings from both committing?",
        topic="Incident response", competency="Incident diagnosis",
    )
    chain = MagicMock()
    chain.invoke.return_value = decision(
        "ASK_FOLLOWUP", 0, question_data=follow_up, missing=["Transaction isolation"], sufficient=False,
    )
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _model: chain
    mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=prompt)
    model = MagicMock()
    model.with_structured_output.return_value = MagicMock()
    mocker.patch.object(nodes, "llm", model)

    update = nodes.adaptive_decision_node(interview)

    assert chain.invoke.call_count == 1
    assert update["decisions"][-1]["action"] == "ASK_FOLLOWUP"
    assert update["decisions"][-1]["recommendation_origin"] == "BOUNDED_REPAIR"
    assert update["pending_adaptive_question"]["question"]["question"] == follow_up["question"]


def test_repair_binds_omitted_source_index_from_current_graph_state(mocker):
    interview = state()
    interview["analyses"][-1] = analysis(
        decision("ADVANCE", 0, sufficient=False, missing=["Transaction isolation"]),
        correctness_score=2, depth_score=2,
    )
    follow_up = question(
        "How would you prevent two simultaneous bookings from both committing?",
        topic="Incident response", competency="Incident diagnosis",
    )
    repair_result = decision("ASK_FOLLOWUP", 0, question_data=follow_up, missing=["Transaction isolation"])
    repair_result["source_plan_index"] = None
    chain = MagicMock()
    chain.invoke.return_value = repair_result
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _model: chain
    mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=prompt)
    model = MagicMock()
    model.with_structured_output.return_value = MagicMock()
    mocker.patch.object(nodes, "llm", model)

    update = nodes.adaptive_decision_node(interview)

    assert update["decisions"][-1]["action"] == "ASK_FOLLOWUP"
    assert update["decisions"][-1]["source_plan_index"] == 0
    assert update["decisions"][-1]["recommendation_status"] == "VALID"
    assert update["pending_adaptive_question"]["source_plan_index"] == 0


def test_incomplete_embedded_followup_uses_active_question_metadata():
    interview = state()
    interview["analyses"][-1] = analysis({
        "action": "ASK_FOLLOWUP",
        "rationale": "Probe the unresolved diagnostic step.",
        "missing_concepts": ["Root-cause isolation"],
        "competency_sufficient": False,
        "source_plan_index": 0,
        "follow_up_is_duplicate": False,
        "follow_up_question": {
            "question": "Which signal would you inspect first to isolate the failure?",
            "topic": "Incident response",
            "primary_competency": "Incident diagnosis",
            "difficulty": "Medium",
        },
    })

    update = nodes.adaptive_decision_node(interview)

    follow_up = update["pending_adaptive_question"]["question"]
    assert follow_up["question_type"] == "Technical"
    assert follow_up["expected_concepts"] == ["Root-cause isolation"]
    assert follow_up["primary_competency"] == "Incident diagnosis"
    assert update["decisions"][-1]["action"] == "ASK_FOLLOWUP"
    assert update["decisions"][-1]["recommendation_status"] == "VALID"
    assert update["decisions"][-1]["recommendation_origin"] == "MODEL"
    assert update["decisions"][-1]["failure_context"]["source_metadata_defaults_applied"] == [
        "expected_concepts", "question_type",
    ]


@pytest.mark.parametrize(
    "failure, expected_kind, expected_exception",
    [
        ("parser", "MALFORMED_REPAIR_OUTPUT", "OutputParserException"),
        ("schema", "ADAPTIVE_SCHEMA_FAILURE", "ValidationError"),
        ("provider", "ADAPTIVE_PROVIDER_FAILURE", "APIConnectionError"),
    ],
)
def test_adaptive_repair_records_specific_failure_classification(
    mocker, failure, expected_kind, expected_exception,
):
    from groq import APIConnectionError
    from httpx import Request
    from langchain_core.exceptions import OutputParserException

    interview = state()
    interview["analyses"][-1] = analysis(
        decision("ADVANCE", 0, sufficient=False, missing=["Transaction isolation"]),
        correctness_score=2, depth_score=2,
    )
    chain = MagicMock()
    if failure == "parser":
        chain.invoke.side_effect = OutputParserException("malformed structured output")
    elif failure == "schema":
        chain.invoke.return_value = {}
    else:
        chain.invoke.side_effect = APIConnectionError(
            message="Connection error", request=Request("POST", "https://api.groq.com")
        )
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _model: chain
    mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=prompt)
    model = MagicMock()
    model.with_structured_output.return_value = MagicMock()
    mocker.patch.object(nodes, "llm", model)

    update = nodes.adaptive_decision_node(interview)

    decision_record = update["decisions"][-1]
    assert decision_record["action"] == "ADVANCE"
    assert decision_record["recommendation_status"] == "FAILURE"
    assert decision_record["failure_kind"] == expected_kind
    assert decision_record["failure_context"]["exception_type"] == expected_exception


def test_analysis_provider_failure_is_not_recorded_as_a_valid_advance():
    interview = state()
    interview["analyses"][-1] = analysis(
        None, status="failed", answer_classification="analysis_failed",
        adaptive_decision={"failure_kind": "provider_error"},
    )

    update = nodes.adaptive_decision_node(interview)

    assert update["decisions"][-1]["action"] == "ADVANCE"
    assert update["decisions"][-1]["recommendation_status"] == "FAILURE"
    assert update["decisions"][-1]["failure_kind"] == "provider_error"
    assert update["adaptation_history"][-1]["recommendation_status"] == "FAILURE"


@pytest.mark.parametrize(
    "result,failure_kind",
    [(RuntimeError("provider unavailable"), "provider_error"), ({}, "malformed_structured_output")],
)
def test_answer_analysis_records_provider_and_schema_failures_separately(mocker, result, failure_kind):
    interview = state()
    interview["current_answer"] = "I checked the service logs."
    chain = MagicMock()
    if isinstance(result, Exception):
        chain.invoke.side_effect = result
    else:
        chain.invoke.return_value = result
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _model: chain
    mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=prompt)
    model = MagicMock()
    mocker.patch.object(nodes, "llm", model)

    update = nodes.analyze_answer_node(interview)

    assert update["analyses"][-1]["status"] == "failed"
    assert update["analyses"][-1]["adaptive_decision"]["failure_kind"] == failure_kind


def test_strong_answer_with_specific_unresolved_gap_may_receive_focused_follow_up():
    interview = state()
    interview["analyses"][-1] = analysis(
        decision("ASK_FOLLOWUP", 0, question_data=question(
            "What signal confirmed the rollback restored service?", topic="Incident response", competency="Incident diagnosis"),
            missing=["Recovery verification"]),
        correctness_score=9, depth_score=8, communication_score=9,
    )
    update = nodes.adaptive_decision_node(interview)
    assert update["decisions"][-1]["action"] == "ASK_FOLLOWUP"
    assert update["decisions"][-1]["missing_concepts"] == ["Recovery verification"]


def test_meaningful_follow_up_is_extra_turn_and_next_plan_item_is_preserved():
    interview = state()
    followup = question("What evidence confirmed that the rollback resolved the outage?", topic="Incident response", competency="Incident diagnosis")
    interview["analyses"][-1] = analysis(decision("ASK_FOLLOWUP", 0, question_data=followup, missing=["Verification"]))
    update = nodes.adaptive_decision_node(interview)
    assert update["pending_adaptive_question"]["source_plan_index"] == 0
    assert update["follow_up_counts"] == {"0": 1}
    assert update["decisions"][-1]["action"] == "ASK_FOLLOWUP"

    followup_turn = nodes.generate_question_node({**interview, **update})
    assert followup_turn["current_question"] == followup["question"]
    assert followup_turn["current_plan_index"] == 0
    assert followup_turn["current_follow_up_for_plan_index"] == 0
    assert followup_turn["plan_cursor"] == 1
    assert followup_turn["remaining_question_count"] == 2
    assert followup_turn["question_metadata"][-1]["difficulty"] == "Medium"
    assert interview["interview_plan"][1]["question"] == "Describe a time you resolved a team disagreement."

    answered_followup = {
        **interview, **followup_turn,
        "answers": interview["answers"] + ["The error rate returned to baseline after rollback."],
        "analyses": interview["analyses"] + [analysis(decision("ADVANCE", 0, sufficient=True))],
        "current_follow_up_for_plan_index": 0,
    }
    advance = nodes.adaptive_decision_node(answered_followup)
    next_question = nodes.generate_question_node({**answered_followup, **advance})
    assert next_question["current_question"] == interview["interview_plan"][1]["question"]
    assert next_question["current_plan_index"] == 1
    assert next_question["plan_cursor"] == 2


@pytest.mark.parametrize("subtype", ["HR_MOTIVATION", "EXPERIENCE_BASED", "SITUATIONAL"])
def test_behavioral_follow_up_preserves_category_subtype_competency_and_difficulty(subtype):
    interview = state()
    planned = interview["interview_plan"][1]
    planned["behavioral_category"] = subtype
    planned["difficulty"] = "Hard"
    interview.update({
        "plan_cursor": 2, "current_plan_index": 1, "current_question_type": "Behavioral",
        "current_behavioral_category": subtype, "current_question_topic": planned["topic"],
        "current_primary_competency": planned["primary_competency"], "current_question_difficulty": "Hard",
        "question_metadata": interview["question_metadata"] + [{**planned, "plan_index": 1, "source_plan_index": 1, "is_follow_up": False}],
        "questions": interview["questions"] + [planned["question"]],
        "answers": interview["answers"] + ["I did nothing."],
        "analyses": interview["analyses"] + [analysis(decision("ASK_FOLLOWUP", 1, question_data=question(
            "Why did you choose not to act, and what would you consider now?", "Behavioral", planned["topic"],
            subtype=subtype, competency=planned["primary_competency"],
        ), missing=["Reasoning"]))],
    })
    update = nodes.adaptive_decision_node(interview)
    asked = nodes.generate_question_node({**interview, **update})
    assert asked["current_question_type"] == "Behavioral"
    assert asked["current_behavioral_category"] == subtype
    assert asked["current_primary_competency"] == planned["primary_competency"]
    assert asked["current_question_difficulty"] == "Hard"
    assert asked["current_follow_up_for_plan_index"] == 1


def test_invalid_and_duplicate_follow_ups_safely_advance():
    interview = state()
    invalid = question("Design a distributed cache.", "Behavioral", "Architecture", competency="System design")
    interview["analyses"][-1] = analysis(decision("ASK_FOLLOWUP", 0, question_data=invalid))
    update = nodes.adaptive_decision_node(interview)
    assert update["decisions"][-1]["action"] == "ADVANCE"
    assert update["decisions"][-1]["recommendation_status"] == "REJECTED"
    assert update["decisions"][-1]["failure_kind"] == "FOLLOW_UP_VALIDATION_FAILURE"
    assert update["pending_adaptive_question"] is None

    repeated = state()
    repeated["analyses"][-1] = analysis(decision("ASK_FOLLOWUP", 0, question_data=repeated["interview_plan"][0]))
    update = nodes.adaptive_decision_node(repeated)
    assert update["decisions"][-1]["action"] == "ADVANCE"
    assert update["decisions"][-1]["recommendation_status"] == "REJECTED"
    assert update["decisions"][-1]["failure_kind"] == "FOLLOW_UP_DUPLICATE"
    assert update["pending_adaptive_question"] is None


def test_safety_ceiling_is_hidden_and_forces_advance():
    interview = state()
    interview["follow_up_counts"] = {"0": nodes.MAX_FOLLOW_UPS_PER_SOURCE}
    proposed = question("What monitoring signal would confirm recovery?", topic="Incident response", competency="Incident diagnosis")
    interview["analyses"][-1] = analysis(decision("ASK_FOLLOWUP", 0, question_data=proposed))
    update = nodes.adaptive_decision_node(interview)
    assert update["decisions"][-1]["action"] == "ADVANCE"
    assert update["decisions"][-1]["failure_kind"] == "FOLLOW_UP_LIMIT_REACHED"
    assert update["pending_adaptive_question"] is None
    assert "max_follow" not in interview["config"]


def test_interview_wide_follow_up_limit_forces_advance():
    interview = state()
    interview["follow_up_counts"] = {"0": 2, "1": 1}
    proposed = question("What monitoring signal would confirm recovery?", topic="Incident response", competency="Incident diagnosis")
    interview["analyses"][-1] = analysis(decision("ASK_FOLLOWUP", 0, question_data=proposed))

    update = nodes.adaptive_decision_node(interview)

    assert nodes.MAX_FOLLOW_UPS_PER_INTERVIEW == 3
    assert update["decisions"][-1]["action"] == "ADVANCE"
    assert update["decisions"][-1]["failure_kind"] == "FOLLOW_UP_LIMIT_REACHED"
    assert update["pending_adaptive_question"] is None


def test_second_consecutive_unknown_answer_advances_without_another_probe():
    interview = state()
    follow_up = question("What is one transaction safeguard?", topic="Incident response", competency="Incident diagnosis")
    interview["questions"].append(follow_up["question"])
    interview["question_metadata"].append({
        **follow_up, "plan_index": 0, "source_plan_index": 0,
        "is_follow_up": True, "follow_up_for_plan_index": 0,
    })
    interview["answers"] = ["I don't know", "I don't know"]
    interview["analyses"] = [
        analysis(None, answer_classification="admitted_unknown", correctness_score=0, depth_score=0),
        analysis(decision("ASK_FOLLOWUP", 0), answer_classification="admitted_unknown", correctness_score=0, depth_score=0),
    ]
    interview.update({
        "current_question": follow_up["question"],
        "current_follow_up_for_plan_index": 0,
        "follow_up_counts": {"0": 1},
        "current_turn_intent": {"intent": "ADMITTED_UNKNOWN", "confidence": 1.0},
    })

    update = nodes.adaptive_decision_node(interview)

    assert update["decisions"][-1]["action"] == "ADVANCE"
    assert update["decisions"][-1]["recommendation_status"] == "POLICY"
    assert update["decisions"][-1]["recommendation_origin"] == "POLICY"
    assert update["pending_adaptive_question"] is None


def test_incomplete_nested_follow_up_is_repaired_without_warning(mocker, caplog):
    interview = state()
    interview["analyses"][-1] = analysis({
        "action": "ASK_FOLLOWUP", "rationale": "Probe pagination detail",
        "missing_concepts": ["cursor validation"], "competency_sufficient": False,
        "source_plan_index": 0, "follow_up_is_duplicate": False,
        "follow_up_question": {
            "topic": "Incident response", "primary_competency": "Incident diagnosis",
            "difficulty": "Medium", "question_type": "Technical",
            "behavioral_category": None, "expected_concepts": ["Validation"],
        },
    }, missing_concepts=["cursor validation"])
    repaired_question = question(
        "How would you validate a cursor before using it?",
        topic="Incident response", competency="Incident diagnosis",
    )
    chain = MagicMock()
    chain.invoke.return_value = decision("ASK_FOLLOWUP", 0, question_data=repaired_question)
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _model: chain
    mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=prompt)
    model = MagicMock()
    model.with_structured_output.return_value = MagicMock()
    mocker.patch.object(nodes, "llm", model)

    with caplog.at_level("WARNING", logger=nodes.logger.name):
        update = nodes.adaptive_decision_node(interview)

    assert update["decisions"][-1]["recommendation_status"] == "VALID"
    assert update["decisions"][-1]["recommendation_origin"] == "BOUNDED_REPAIR"
    assert update["pending_adaptive_question"]["question"]["question"] == repaired_question["question"]
    assert "Adaptive recommendation schema validation failed" not in caplog.text


def test_missing_adaptive_decision_enters_bounded_repair_without_value_error(mocker, caplog):
    interview = state()
    interview["analyses"][-1] = analysis(None)
    chain = MagicMock()
    chain.invoke.return_value = decision("ADVANCE", 0, sufficient=True)
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _model: chain
    mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=prompt)
    model = MagicMock()
    model.with_structured_output.return_value = MagicMock()
    mocker.patch.object(nodes, "llm", model)

    with caplog.at_level("WARNING", logger=nodes.logger.name):
        update = nodes.adaptive_decision_node(interview)

    assert update["decisions"][-1]["action"] == "ADVANCE"
    assert update["decisions"][-1]["recommendation_status"] == "VALID"
    assert update["decisions"][-1]["recommendation_origin"] == "BOUNDED_REPAIR"
    assert "application validation failed" not in caplog.text


def test_repeated_weak_answers_do_not_loop_after_repaired_advance(mocker):
    interview = state()
    interview["analyses"][-1] = analysis(None, answer_classification="irrelevant")
    chain = MagicMock()
    chain.invoke.return_value = decision("ADVANCE", 0, sufficient=True)
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _model: chain
    mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=prompt)
    model = MagicMock()
    model.with_structured_output.return_value = MagicMock()
    mocker.patch.object(nodes, "llm", model)

    update = nodes.adaptive_decision_node(interview)
    assert update["decisions"][-1]["action"] == "ADVANCE"
    assert update["decisions"][-1]["recommendation_status"] == "VALID"
    assert update["decisions"][-1]["recommendation_origin"] == "BOUNDED_REPAIR"
    assert update["pending_adaptive_question"] is None


def test_empty_answer_and_failed_analysis_advance_without_scoring_or_clarification():
    for classification, status in [("blank", "success"), ("analysis_failed", "failed")]:
        interview = state()
        interview["analyses"][-1] = analysis(None, answer_classification=classification, status=status)
        update = nodes.adaptive_decision_node(interview)
        assert update["decisions"][-1]["action"] == "ADVANCE"
        assert update["pending_adaptive_question"] is None
        if classification == "blank":
            assert update["decisions"][-1]["recommendation_status"] == "NOT_APPLICABLE"
            assert "failure_kind" not in update["decisions"][-1]


def test_irrelevant_answer_can_be_clarified_once_when_decision_has_a_specific_gap():
    interview = state()
    clarification = question("Which incident response step would you start with, and why?", topic="Incident response", competency="Incident diagnosis")
    interview["analyses"][-1] = analysis(
        decision("ASK_FOLLOWUP", 0, question_data=clarification, missing=["Diagnostic reasoning"]),
        answer_classification="irrelevant",
    )
    update = nodes.adaptive_decision_node(interview)
    assert update["decisions"][-1]["action"] == "ASK_FOLLOWUP"
    assert update["pending_adaptive_question"]["question"]["question"] == clarification["question"]


def test_model_flagged_duplicate_follow_up_is_rejected():
    interview = state()
    proposed = question("What was the rollback verification signal?", topic="Incident response", competency="Incident diagnosis")
    interview["analyses"][-1] = analysis(decision("ASK_FOLLOWUP", 0, question_data=proposed, duplicate=True))
    update = nodes.adaptive_decision_node(interview)
    assert update["decisions"][-1]["action"] == "ADVANCE"
    assert update["pending_adaptive_question"] is None


@pytest.mark.parametrize("action, cursor, expected", [("FINISH", 1, "ADVANCE"), ("FINISH", 3, "FINISH")])
def test_finish_only_ends_when_confirmed_plan_is_exhausted(action, cursor, expected):
    interview = state()
    interview["plan_cursor"] = cursor
    interview["analyses"][-1] = analysis(decision(action, 0, sufficient=True))
    update = nodes.adaptive_decision_node(interview)
    assert update["decisions"][-1]["action"] == expected
    next_state = {**interview, **update}
    if cursor == 3:
        assert route_after_decision(next_state) == "evaluate"
    else:
        assert route_after_decision(next_state) == "generate_question"


def test_invalid_decision_output_gets_one_bounded_validation_repair(mocker):
    interview = state()
    interview["analyses"][-1]["adaptive_decision"] = {"action": "FOLLOW_UP", "reasoning": "legacy"}
    follow_up = question("Which observation would help isolate the incident cause?", topic="Incident response", competency="Incident diagnosis")
    chain = MagicMock()
    chain.invoke.return_value = decision("ASK_FOLLOWUP", 0, question_data=follow_up, missing=["Verification"])
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _model: chain
    mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=prompt)
    model = MagicMock()
    model.with_structured_output.return_value = MagicMock()
    mocker.patch.object(nodes, "llm", model)

    update = nodes.adaptive_decision_node(interview)
    assert update["decisions"][-1]["action"] == "ASK_FOLLOWUP"
    assert update["decisions"][-1]["recommendation_status"] == "VALID"
    assert update["decisions"][-1]["recommendation_origin"] == "BOUNDED_REPAIR"


def test_analysis_and_decision_share_one_model_invocation(mocker):
    from app.engine.nodes import analyze_answer_node

    interview = state()
    interview.update({
        "current_answer": "I checked logs and verified the rollback.",
        "current_expected_concepts": ["Logs", "Verification"],
        "profile": {"resume_summary": "Backend engineer with production support experience.", "skills": ["Python", "Redis"]},
        "conversation_history": [{
            "question": interview["current_question"], "utterance": "Can I have a hint?",
            "intent": "HINT", "response": "Think about the evidence first.",
        }],
    })
    response = AnswerAnalysis(
        correctness_score=7, depth_score=7, communication_score=8,
        feedback="Good evidence.", is_satisfactory=True,
        adaptive_decision=decision("ADVANCE", 0, sufficient=True),
    )
    chain = mocker.MagicMock()
    chain.invoke.return_value = response
    prompt = mocker.MagicMock()
    prompt.__or__ = lambda _self, _other: chain
    prompt_builder = mocker.patch("app.engine.nodes.ChatPromptTemplate.from_messages", return_value=prompt)
    model = mocker.MagicMock()
    mocker.patch("app.engine.nodes.llm", model)

    update = analyze_answer_node(interview)
    assert chain.invoke.call_count == 1
    assert update["analyses"][0]["adaptive_decision"]["action"] == "ADVANCE"
    invocation = chain.invoke.call_args.args[0]
    assert "I checked logs and rolled back the deployment." in invocation["prior_evidence"]
    assert "Backend engineer" in invocation["candidate_context"]
    assert "requested hint (interaction only; not evidence)" in invocation["conversation_context"]
    policy_text = " ".join(str(message) for message in prompt_builder.call_args.args[0]).lower()
    assert "do not reward length or penalize brevity" in policy_text
    assert "unsupported claims" in policy_text
    assert "meaningful extension" in policy_text


def test_graph_preserves_history_and_plan_cursor_across_follow_up_invocations(mocker):
    interview = state()
    interview.update({"questions": [], "question_metadata": [], "answers": [], "analyses": [], "topics_covered": [], "plan_cursor": 0})
    def fake_analysis(state):
        decision_data = decision("ASK_FOLLOWUP", 0, question_data=question(
            "What evidence showed the rollback worked?", topic="Incident response", competency="Incident diagnosis"))
        return {
            "answers": state.get("answers", []) + [state["current_answer"]], "current_answer": None,
            "analyses": [*state.get("analyses", []), analysis(decision_data)],
        }
    mocker.patch("app.engine.graph.analyze_answer_node", side_effect=fake_analysis)
    mocker.patch("app.engine.graph.classify_turn_node", side_effect=lambda state: {
        "current_turn_intent": {"intent": "ANSWER", "confidence": 1.0},
        "classified_utterance": state["current_input"], "current_input": None,
    })
    mocker.patch("app.engine.graph.adaptive_decision_node", side_effect=nodes.adaptive_decision_node)
    mocker.patch("app.engine.graph.generate_question_node", side_effect=nodes.generate_question_node)
    engine = InterviewEngine()
    first = engine.start_interview(interview)
    assert first["plan_cursor"] == 1

    followup = engine.process_answer(first, "Logs showed a timeout.")
    assert followup["current_follow_up_for_plan_index"] == 0
    assert followup["plan_cursor"] == 1
    assert len(followup["questions"]) == 2
    assert len(followup["interview_plan"]) == interview["config"]["question_count"]

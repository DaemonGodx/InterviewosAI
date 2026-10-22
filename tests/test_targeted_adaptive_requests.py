from unittest.mock import MagicMock

import pytest

from app.engine import nodes
from app.engine.graph import InterviewEngine
from app.models.schemas import AdaptiveDecision, AnswerAnalysis, InterviewQuestion, TurnIntent


def _question(text, difficulty="Medium"):
    return InterviewQuestion(
        question=text,
        topic="Debugging",
        primary_competency="Incident diagnosis",
        difficulty=difficulty,
        question_type="Technical",
        expected_concepts=["Logs", "Verification"],
    ).model_dump(mode="json")


def _state(difficulty="Medium"):
    first = _question("How would you debug an API timeout?", difficulty)
    second = _question("How would you verify a fix?", difficulty)
    return {
        "session_id": "adaptive-request-test",
        "config": {"target_role": "Backend", "experience_level": "Mid-level", "question_count": 2, "interview_type": "Technical"},
        "profile": {}, "interview_plan": [first, second],
        "questions": [first["question"]], "answers": [], "analyses": [], "decisions": [],
        "topics_covered": [first["topic"]],
        "question_metadata": [{**first, "plan_index": 0, "source_plan_index": 0, "is_follow_up": False}],
        "current_question": first["question"], "current_question_type": "Technical",
        "current_question_topic": first["topic"], "current_question_difficulty": difficulty,
        "current_primary_competency": first["primary_competency"],
        "current_expected_concepts": first["expected_concepts"],
        "current_plan_index": 0, "current_follow_up_for_plan_index": None,
        "plan_cursor": 1, "plan_execution_order": [0, 1], "remaining_question_count": 1,
        "follow_up_counts": {}, "adaptation_history": [], "conversation_history": [],
        "status": "IN_PROGRESS",
    }


def _adaptive_analysis(action, follow_up=None, *, duplicate=False):
    return {
        "correctness_score": 0, "depth_score": 0, "communication_score": 5,
        "feedback": "No technical answer was provided.", "is_satisfactory": False,
        "status": "success", "answer_classification": "substantive",
        "adaptive_decision": {
            "action": action,
            "rationale": "Address the candidate's explicit request while preserving the current competency.",
            "missing_concepts": ["Diagnostic evidence"] if action == "ASK_FOLLOWUP" else [],
            "competency_sufficient": action != "ASK_FOLLOWUP",
            "follow_up_question": follow_up,
            "source_plan_index": 0,
            "follow_up_is_duplicate": duplicate,
        },
    }


class _Prompt:
    def __or__(self, model):
        return _Chain(model)


class _Chain:
    def __init__(self, model):
        self.model = model

    def invoke(self, _inputs):
        return self.model.respond()


class _FixtureModel:
    def __init__(self, turn_intent, answer_analysis, response="Consider one piece of evidence that could narrow the cause.", adaptive_decision=None):
        self.turn_intent = turn_intent
        self.answer_analysis = answer_analysis
        self.response = response
        self.adaptive_decision = adaptive_decision

    def with_structured_output(self, schema):
        if schema is TurnIntent:
            return _Structured(self.turn_intent)
        if schema is AnswerAnalysis:
            return _Structured(self.answer_analysis)
        if schema is AdaptiveDecision:
            return _Structured(self.adaptive_decision)
        raise AssertionError(f"Unexpected schema: {schema}")

    def respond(self):
        return self.response


class _Structured:
    def __init__(self, value):
        self.value = value

    def respond(self):
        return self.value


@pytest.mark.parametrize(
    "utterance,requested_action,source_difficulty,followup_difficulty",
    [
        ("I don't know this, ask a bit easy.", "EASIER", "Hard", "Medium"),
        ("I don't know; ask me a follow-up.", "FOLLOW_UP", "Medium", "Medium"),
    ],
)
def test_compound_unknown_request_runs_through_graph_and_selects_valid_followup(
    mocker, utterance, requested_action, source_difficulty, followup_difficulty
):
    state = _state(source_difficulty)
    follow_up = _question("What log evidence would narrow down the timeout source?", followup_difficulty)
    analysis_action = "ADVANCE" if requested_action == "EASIER" else "ASK_FOLLOWUP"
    mocker.patch.object(
        nodes,
        "llm",
        _FixtureModel(
            {"intent": "ADMITTED_UNKNOWN", "confidence": 0.99, "requested_action": requested_action},
            _adaptive_analysis(analysis_action, follow_up if analysis_action == "ASK_FOLLOWUP" else None),
            adaptive_decision=_adaptive_analysis("ASK_FOLLOWUP", follow_up)["adaptive_decision"],
        ),
    )
    mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=_Prompt())

    result = InterviewEngine().process_answer(state, utterance)

    assert result["current_turn_intent"]["intent"] == "ADMITTED_UNKNOWN"
    assert result["current_turn_intent"]["requested_action"] == requested_action
    assert result["analyses"][-1]["answer_classification"] == "admitted_unknown"
    assert result["decisions"][-1]["action"] == "ASK_FOLLOWUP"
    if requested_action == "EASIER":
        assert result["decisions"][-1]["recommendation_origin"] == "BOUNDED_REPAIR"
    assert result["current_question"] == follow_up["question"]
    assert result["current_question_difficulty"] == followup_difficulty
    if requested_action == "EASIER":
        assert result["question_metadata"][-1]["difficulty_override"] == followup_difficulty
        assert result["next_difficulty_override"] is None
    assert result["current_primary_competency"] == "Incident diagnosis"
    assert result["current_follow_up_for_plan_index"] == 0
    assert result["plan_cursor"] == 1
    assert result["interview_plan"][1]["question"] == state["interview_plan"][1]["question"]


def test_plain_unknown_gets_one_bounded_same_competency_followup(mocker):
    state = _state()
    follow_up = _question("What observation would you check first to narrow down an API timeout?", "Medium")
    mocker.patch.object(
        nodes,
        "llm",
        _FixtureModel(
            {"intent": "ADMITTED_UNKNOWN", "confidence": 0.99},
            _adaptive_analysis("ADVANCE"),
            adaptive_decision=_adaptive_analysis("ASK_FOLLOWUP", follow_up)["adaptive_decision"],
        ),
    )
    mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=_Prompt())

    result = InterviewEngine().process_answer(state, "I don't know.")

    assert result["current_turn_intent"]["intent"] == "ADMITTED_UNKNOWN"
    assert result["current_turn_intent"]["requested_action"] is None
    assert result["analyses"][-1]["answer_classification"] == "admitted_unknown"
    assert result["decisions"][-1]["action"] == "ASK_FOLLOWUP"
    assert result["decisions"][-1]["recommendation_origin"] == "BOUNDED_REPAIR"
    assert result["current_question"] == follow_up["question"]
    assert result["current_question_difficulty"] == "Medium"
    assert result["plan_cursor"] == state["plan_cursor"]
    assert result["current_follow_up_for_plan_index"] == 0


def test_clear_easier_request_survives_ambiguous_primary_classification(mocker):
    chain = MagicMock()
    chain.invoke.return_value = {"intent": "AMBIGUOUS", "confidence": 0.4}
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _model: chain
    mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=prompt)
    mocker.patch.object(nodes, "llm", MagicMock())

    result = nodes.classify_turn_node({
        "current_input": "Can you ask me an easier question?",
        "current_question": "How would you debug an API timeout?",
    })

    assert result["current_turn_intent"]["intent"] == "ADMITTED_UNKNOWN"
    assert result["current_turn_intent"]["requested_action"] == "EASIER"


def test_plain_unknown_after_one_probe_does_not_force_another(mocker):
    state = _state()
    state["follow_up_counts"] = {"0": 1}
    state["question_metadata"].append({
        **state["question_metadata"][0], "question": "What evidence narrows the timeout?",
        "is_follow_up": True, "follow_up_for_plan_index": 0,
    })
    state["questions"].append("What evidence narrows the timeout?")
    state["current_turn_intent"] = {"intent": "ADMITTED_UNKNOWN"}
    state["analyses"] = [_adaptive_analysis("ADVANCE")]
    mocker.patch.object(nodes, "llm", MagicMock())

    update = nodes.adaptive_decision_node(state)

    assert update["decisions"][-1]["action"] == "ADVANCE"
    assert update["decisions"][-1]["recommendation_status"] == "VALID"
    assert update["pending_adaptive_question"] is None


def test_unknown_plus_hint_uses_non_scoring_hint_path_and_keeps_question(mocker):
    from app.engine import graph

    state = _state()
    utterance = "I don't know; can you give me a hint?"

    def classify(_state):
        return {
            "current_turn_intent": {"intent": "ADMITTED_UNKNOWN", "requested_action": "HINT", "confidence": 0.99},
            "current_input": None,
            "classified_utterance": utterance,
        }

    mocker.patch("app.engine.graph.classify_turn_node", side_effect=classify)
    mocker.patch.object(nodes, "llm", MagicMock())
    chain = MagicMock()
    chain.invoke.return_value = "Consider what evidence along the request path could locate the delay."
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _model: chain
    mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=prompt)

    result = InterviewEngine().process_answer(state, utterance)

    assert result["current_question"] == state["current_question"]
    assert result["answers"] == result["analyses"] == []
    assert result["plan_cursor"] == state["plan_cursor"]
    assert result["interviewer_response"] == chain.invoke.return_value
    assert result["conversation_history"][-1]["intent"] == "HINT"
    assert result["conversation_history"][-1]["underlying_intent"] == "ADMITTED_UNKNOWN"


def test_easier_followup_rejects_unsupported_or_not_lower_difficulty():
    state = _state("Medium")
    state["current_turn_intent"] = {"intent": "ADMITTED_UNKNOWN", "requested_action": "EASIER"}
    state["analyses"] = [_adaptive_analysis("ASK_FOLLOWUP", _question("Clarify the diagnosis?", "Easier"))]

    result = nodes.adaptive_decision_node(state)

    assert result["decisions"][-1]["action"] == "ADVANCE"
    assert result["pending_adaptive_question"] is None


def test_easier_request_at_lowest_difficulty_falls_back_safely():
    state = _state("Easy")
    state["current_turn_intent"] = {"intent": "ADMITTED_UNKNOWN", "requested_action": "EASIER"}
    state["analyses"] = [_adaptive_analysis("ASK_FOLLOWUP", _question("A simpler diagnostic prompt?", "Easy"))]

    result = nodes.adaptive_decision_node(state)

    assert result["decisions"][-1]["action"] == "ADVANCE"
    assert result["pending_adaptive_question"] is None


def test_easier_request_at_lowest_falls_back_to_hint_without_advancing(mocker):
    state = _state("Easy")
    mocker.patch.object(
        nodes,
        "llm",
        _FixtureModel(
            {"intent": "ADMITTED_UNKNOWN", "confidence": 0.99, "requested_action": "EASIER"},
            _adaptive_analysis("ADVANCE"),
        ),
    )
    mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=_Prompt())

    result = InterviewEngine().process_answer(state, "I don't know; ask me an easier question.")

    assert result["current_question"] == state["current_question"]
    assert result["current_question_difficulty"] == "Easy"
    assert result["plan_cursor"] == state["plan_cursor"]
    assert result["answers"] == result["analyses"] == []
    assert result["interviewer_response"]
    assert result["conversation_history"][-1]["intent"] == "HINT"
    assert result["conversation_history"][-1]["requested_action"] == "EASIER"


def test_followup_request_field_is_backward_compatible():
    old_payload = {"intent": "ADMITTED_UNKNOWN", "confidence": 0.9}

    parsed = TurnIntent.model_validate(old_payload)

    assert parsed.requested_action is None


@pytest.mark.parametrize("utterance", ["i dont know", "I don't know.", "I do not know!"])
def test_plain_admission_is_classified_without_the_llm(mocker, utterance):
    model = MagicMock()
    mocker.patch.object(nodes, "llm", model)

    result = nodes.classify_turn_node({
        "current_input": utterance,
        "current_question": "How would you debug an API timeout?",
    })

    assert result["current_turn_intent"]["intent"] == "ADMITTED_UNKNOWN"
    assert result["current_turn_intent"]["confidence"] == 1.0
    model.with_structured_output.assert_not_called()


def test_technical_uncertainty_with_evidence_still_uses_classifier(mocker):
    chain = MagicMock()
    chain.invoke.return_value = {"intent": "ANSWER", "confidence": 0.94}
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _model: chain
    mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=prompt)
    model = MagicMock()
    mocker.patch.object(nodes, "llm", model)

    result = nodes.classify_turn_node({
        "current_input": "I don't know for sure, but I would inspect the logs and traces.",
        "current_question": "How would you debug an API timeout?",
    })

    assert result["current_turn_intent"]["intent"] == "ANSWER"
    model.with_structured_output.assert_called_once_with(TurnIntent)


def test_classifier_recovers_obvious_compound_easier_request_when_model_omits_field(mocker):
    chain = MagicMock()
    chain.invoke.return_value = {"intent": "ADMITTED_UNKNOWN", "confidence": 0.95}
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _model: chain
    mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=prompt)
    mocker.patch.object(nodes, "llm", MagicMock())

    result = nodes.classify_turn_node({
        "current_input": "I don't know this, ask a bit easy.",
        "current_question": "How would you debug an API timeout?",
    })

    assert result["current_turn_intent"]["intent"] == "ADMITTED_UNKNOWN"
    assert result["current_turn_intent"]["requested_action"] == "EASIER"


def test_answer_analysis_receives_the_explicit_request_as_policy_context(mocker):
    state = _state("Hard")
    state.update({
        "current_answer": "I don't know; ask me an easier question.",
        "current_turn_intent": {"intent": "ADMITTED_UNKNOWN", "requested_action": "EASIER"},
    })
    chain = MagicMock()
    chain.invoke.return_value = {
        "correctness_score": 0, "depth_score": 0, "communication_score": 5,
        "feedback": "No technical answer was provided.", "is_satisfactory": False,
        "status": "success", "answer_classification": "substantive",
        "adaptive_decision": None,
    }
    prompt = MagicMock()
    prompt.__or__ = lambda _self, _model: chain
    prompt_builder = mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=prompt)
    model = MagicMock()
    mocker.patch.object(nodes, "llm", model)

    nodes.analyze_answer_node(state)

    assert chain.invoke.call_args.args[0]["requested_action_instruction"] == "EASIER"
    system_prompt = " ".join(str(message) for message in prompt_builder.call_args.args[0])
    assert "explicit easier request" in system_prompt.lower()


def test_incorrect_answer_can_take_a_recommended_relevant_followup():
    state = _state()
    follow_up = _question("What evidence would you check first to locate the timeout?", "Medium")
    analysis = _adaptive_analysis("ASK_FOLLOWUP", follow_up)
    analysis.update({"correctness_score": 0, "answer_classification": "irrelevant"})
    state["analyses"] = [analysis]

    update = nodes.adaptive_decision_node(state)

    assert update["decisions"][-1]["action"] == "ASK_FOLLOWUP"
    assert update["pending_adaptive_question"]["question"]["question"] == follow_up["question"]


@pytest.mark.parametrize("mode", ["Text", "Voice"])
def test_cli_treats_unknown_plus_hint_as_conversation_without_analysis(mocker, capsys, mode):
    from app import __main__ as cli

    question = "How would you debug an API timeout?"
    start = {
        "config": {"interview_mode": mode, "question_count": 1},
        "questions": [question], "question_metadata": [{"is_follow_up": False}],
        "current_question": question, "current_difficulty": "Medium",
        "current_question_difficulty": "Medium", "topics_covered": ["Debugging"],
        "remaining_question_count": 0, "status": "IN_PROGRESS", "analyses": [],
    }
    calls = []

    class FakeEngine:
        def start_interview(self, _initial):
            return dict(start)

        def process_answer(self, current, answer):
            calls.append((answer, current.get("analyses", [])))
            if len(calls) == 1:
                return {
                    **current,
                    "current_turn_intent": {"intent": "ADMITTED_UNKNOWN", "requested_action": "HINT"},
                    "interviewer_response": "Consider what observation could narrow the source of the delay.",
                    "conversation_history": [{"intent": "HINT", "underlying_intent": "ADMITTED_UNKNOWN"}],
                }
            return {
                **current, "current_turn_intent": {"intent": "ANSWER"},
                "analyses": [{"status": "success", "correctness_score": 7, "feedback": "Clear."}],
                "status": "COMPLETED",
            }

    class FakeDB:
        def save_session(self, _state):
            return None

    mocker.patch.object(cli, "InterviewEngine", FakeEngine)
    if mode == "Text":
        mocker.patch.object(cli.Prompt, "ask", side_effect=["I don't know; hint, please.", "My answer"])
    else:
        mocker.patch.object(cli, "AudioService", return_value=MagicMock())
        speech = MagicMock()
        mocker.patch.object(cli, "SpeechService", return_value=speech)
        mocker.patch.object(cli, "get_voice_answer", side_effect=["I don't know; hint, please.", "My answer"])

    cli.run_interview_loop(FakeDB(), start)

    assert calls[1][1] == []
    assert "Consider what observation" in capsys.readouterr().out
    if mode == "Voice":
        assert speech.speak.call_count == 2

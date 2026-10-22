from unittest.mock import MagicMock

import pytest

from app.engine.graph import InterviewEngine
from app.engine import nodes
from app.models.schemas import (
    AdaptiveDecision, AnswerAnalysis, CandidateProfile, InterviewConfig,
    InterviewQuestion, InterviewState,
)
from app.services.planner_service import PlannerService


def planned_question(category, number):
    if category == "Behavioral":
        text, topic, concepts = (
            f"Describe a time you demonstrated ownership, example {number}.",
            "Ownership", ["Personal contribution", "Outcome"],
        )
    else:
        text, topic, concepts = (
            f"How would you debug a production API issue, example {number}?",
            "Debugging", ["Logs", "Verification"],
        )
    return InterviewQuestion(
        question=text, topic=topic, primary_competency=topic,
        difficulty="Medium", question_type=category, expected_concepts=concepts,
    )


@pytest.mark.parametrize("interview_type", ["Technical", "Behavioral", "Mixed"])
def test_interview_types_run_through_graph_to_evaluation(mocker, interview_type):
    count = 2
    categories = PlannerService.expected_types(interview_type, count)
    plan = [planned_question(category, index + 1) for index, category in enumerate(categories)]
    config = InterviewConfig(
        target_role="Backend Engineer", experience_level="Mid-level",
        interview_type=interview_type, question_count=count,
    )
    initial = InterviewState(
        session_id=f"e2e-{interview_type.lower()}", config=config,
        profile=CandidateProfile(name="Test Candidate"), interview_plan=plan,
    ).model_dump(mode="json")

    responses = {
        AnswerAnalysis: [AnswerAnalysis(
            correctness_score=8, depth_score=7, communication_score=8,
            feedback="The response gives relevant evidence and reasoning.",
            is_satisfactory=True,
            adaptive_decision={
                "action": "ADVANCE", "rationale": "Sufficient evidence; continue the confirmed plan.",
                "competency_sufficient": True, "source_plan_index": index,
            },
        ) for index in range(count)],
    }

    def structured_output(schema):
        chain = MagicMock()
        chain.invoke.side_effect = lambda _input: responses[schema].pop(0)
        return chain

    model = MagicMock()
    model.with_structured_output.side_effect = structured_output
    mocker.patch.object(nodes, "llm", model)
    mocker.patch("app.engine.graph.classify_turn_node", side_effect=lambda state: {
        "current_turn_intent": {"intent": "ANSWER", "confidence": 1.0},
        "classified_utterance": state["current_input"], "current_input": None,
    })
    prompt = MagicMock()
    prompt.__or__ = lambda _self, other: other
    mocker.patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=prompt)

    engine = InterviewEngine()
    state = engine.start_interview(initial)
    asked_categories = []
    for index in range(count):
        asked_categories.append(state["current_question_type"])
        state = engine.process_answer(state, f"Specific answer with relevant evidence number {index + 1}.")

    assert asked_categories == categories
    assert len(state["questions"]) == len(state["answers"]) == len(state["analyses"]) == count
    assert state["status"] == "COMPLETED"
    assert state["evaluation"]["overall_score"] == pytest.approx(77.0)
    if interview_type == "Technical":
        assert state["evaluation"]["technical_score"] == 8.0
        assert state["evaluation"]["behavioral_score"] is None
    elif interview_type == "Behavioral":
        assert state["evaluation"]["technical_score"] is None
        assert state["evaluation"]["behavioral_score"] == 7.7
    else:
        assert state["evaluation"]["technical_score"] == 8.0
        assert state["evaluation"]["behavioral_score"] == 7.7

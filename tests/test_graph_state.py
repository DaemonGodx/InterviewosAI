from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.engine.state import normalize_graph_state
from app.models.schemas import (
    AdaptiveDecision, AnswerAnalysis, CandidateProfile, EvaluationReport,
    InterviewConfig, InterviewQuestion, InterviewState,
)


def test_graph_boundary_normalizes_pydantic_state_to_json_compatible_dicts():
    interview = InterviewState(
        session_id="typed-state",
        config=InterviewConfig(target_role="Engineer", experience_level="Junior", question_count=1),
        profile=CandidateProfile(name="Candidate"),
        interview_plan=[InterviewQuestion(
            question="How do you test a change?", topic="Testing", difficulty="Easy",
            question_type="Technical", expected_concepts=["Verification"],
        )],
        analyses=[AnswerAnalysis(
            correctness_score=6, depth_score=5, communication_score=7,
            feedback="Partial", is_satisfactory=False,
        )],
        decisions=[{"action": "ADVANCE", "rationale": "Continue", "competency_sufficient": True}],
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        evaluation=EvaluationReport(
            overall_score=None, technical_score=None, problem_solving_score=None,
            communication_score=None, behavioral_score=None, evaluation_confidence="Very Low",
            resume_claims=[], interview_demonstrated_strengths=[], weaknesses=[], technical_gaps=[],
            recommended_topics=[], role_readiness="Insufficient evidence", final_feedback="No evidence.",
        ),
    )

    graph_state = normalize_graph_state(interview)

    assert isinstance(graph_state["config"], dict)
    assert isinstance(graph_state["profile"], dict)
    assert isinstance(graph_state["interview_plan"][0], dict)
    assert isinstance(graph_state["analyses"][0], dict)
    assert isinstance(graph_state["decisions"][0], dict)
    assert isinstance(graph_state["evaluation"], dict)
    assert graph_state["started_at"].endswith("Z")


def test_graph_boundary_rejects_invalid_config_and_nested_models():
    state = {
        "session_id": "invalid-state",
        "config": {"target_role": "Engineer", "experience_level": "Junior", "question_count": 0},
        "profile": {}, "interview_plan": [], "analyses": [], "decisions": [],
    }
    with pytest.raises(ValidationError):
        normalize_graph_state(state)

    state["config"]["question_count"] = 1
    # Decision history is preserved as historical JSON; current recommendations
    # are validated at the adaptive node rather than on old persisted records.
    state["decisions"] = [{"action": "FOLLOW_UP", "reasoning": "legacy record", "topic": None}]
    assert normalize_graph_state(state)["decisions"][0]["action"] == "FOLLOW_UP"

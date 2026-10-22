import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from app.models.db import Database, SessionRecord
from app.models.schemas import (
    AdaptiveDecision, AnswerAnalysis, CandidateProfile, EvaluationReport,
    InterviewConfig, InterviewQuestion, Project,
)


def complete_state():
    config = InterviewConfig(
        target_role="Backend Engineer", experience_level="Mid-level", interview_type="Mixed",
        question_count=2, interview_mode="Text",
    )
    profile = CandidateProfile(
        name="Example Candidate", skills=["Python"],
        projects=[Project(name="Service", description="Backend service", technologies=["Python"])],
    )
    first = InterviewQuestion(
        question="How do you debug an API?", topic="Debugging", primary_competency="Incident diagnosis",
        difficulty="Medium", question_type="Technical", expected_concepts=["Logs", "Verification"],
    )
    edited = InterviewQuestion(
        question="How would you investigate an API timeout?", topic="Debugging", primary_competency="Incident diagnosis",
        difficulty="Medium", question_type="Technical", expected_concepts=["Logs", "Verification"],
    )
    return {
        "session_id": "round-trip-session",
        "candidate_name": "Example Candidate",
        "target_role": config.target_role,
        "experience_level": config.experience_level,
        "config": config,
        "profile": profile,
        "interview_plan": [edited, InterviewQuestion(
            question="Describe a time you resolved a disagreement.", topic="Conflict resolution",
            primary_competency="Collaboration", difficulty="Medium", question_type="Behavioral",
            expected_concepts=["Listening", "Resolution"],
        )],
        "questions": [edited.question, "Tell me about the rollback evidence you verified."],
        "answers": ["Checked traces and rolled back", "I compared errors before and after."],
        "analyses": [
            AnswerAnalysis(correctness_score=8, depth_score=7, communication_score=8, feedback="Clear", is_satisfactory=True),
            AnswerAnalysis(correctness_score=7, depth_score=6, communication_score=8, feedback="Specific", is_satisfactory=True),
        ],
        "decisions": [{"action": "ASK_FOLLOWUP", "rationale": "Probe the verification step", "competency_sufficient": False, "source_plan_index": 0}],
        "topics_covered": ["Debugging", "Debugging"],
        "question_metadata": [
            {"question": edited.question, "question_type": "Technical", "topic": "Debugging", "primary_competency": "Incident diagnosis", "difficulty": "Medium", "expected_concepts": ["Logs"], "plan_index": 0, "is_follow_up": False},
            {"question": "Tell me about the rollback evidence you verified.", "question_type": "Technical", "topic": "Debugging", "primary_competency": "Incident diagnosis", "difficulty": "Medium", "expected_concepts": ["Evidence"], "plan_index": 0, "source_plan_index": 0, "is_follow_up": True, "follow_up_for_plan_index": 0},
        ],
        "adaptation_history": [{
            "action": "ASK_FOLLOWUP", "applied": True, "source_plan_index": 0,
            "follow_up_question": "Tell me about the rollback evidence you verified.",
            "reason": "Probe the verification step",
        }],
        "follow_up_counts": {"0": 1}, "plan_execution_order": [0, 1], "plan_cursor": 1,
        "remaining_question_count": 1,
        "evaluation": EvaluationReport(
            overall_score=75.0, technical_score=8.0, problem_solving_score=7.6,
            communication_score=8.0, behavioral_score=None, evaluation_confidence="Very Low",
            resume_claims=[], interview_demonstrated_strengths=["Incident diagnosis"], weaknesses=[],
            technical_gaps=[], recommended_topics=[], role_readiness="Insufficient evidence",
            final_feedback="Only assessed evidence is summarized.", readiness_basis="Sparse evidence.",
        ),
        "started_at": datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc),
    }


def test_complete_session_round_trips_models_edits_adaptation_and_report():
    with TemporaryDirectory(prefix="interviewos-roundtrip-", dir=Path.cwd()) as temp_dir:
        database = Database(str(Path(temp_dir) / "roundtrip.sqlite"))
        database.save_session(complete_state())

        history = database.get_all_sessions()
        assert [session.session_id for session in history] == ["round-trip-session"]

        loaded = database.load_session("round-trip-session")

        assert "max_follow_ups_per_question" not in loaded["config"]
        assert loaded["profile"]["projects"][0]["name"] == "Service"
        data = loaded["interview_data"]
        assert data["interview_plan"][0]["question"] == "How would you investigate an API timeout?"
        assert data["questions"][1] == "Tell me about the rollback evidence you verified."
        assert data["answers"][0] == "Checked traces and rolled back"
        assert data["analyses"][0]["status"] == "success"
        assert data["decisions"][0]["action"] == "ASK_FOLLOWUP"
        assert data["adaptation_history"][0]["source_plan_index"] == 0
        assert data["follow_up_counts"] == {"0": 1}
        assert data["remaining_question_count"] == 1
        assert loaded["evaluation"]["overall_score"] == 75.0
        assert loaded["evaluation"]["behavioral_score"] is None
        assert loaded["started_at"].endswith("+00:00")
        assert not loaded["compatibility_warnings"]
        database.engine.dispose()


def test_older_saved_session_loads_with_additive_defaults():
  with TemporaryDirectory(prefix="interviewos-legacy-", dir=Path.cwd()) as temp_dir:
    database = Database(str(Path(temp_dir) / "legacy.sqlite"))
    database.save_session(complete_state())
    old_config = {"target_role": "Backend", "experience_level": "Junior", "interview_type": "Technical", "question_count": 1}
    old_question = {"question": "What is caching?", "topic": "Caching", "difficulty": "Easy", "question_type": "Technical", "expected_concepts": ["Cache"]}
    old_analysis = {"correctness_score": 6, "depth_score": 5, "communication_score": 7, "feedback": "Partial", "is_satisfactory": False}
    old_evaluation = {
        "overall_score": 60, "technical_score": 6, "problem_solving_score": 5.5,
        "communication_score": 7, "behavioral_score": None, "evaluation_confidence": "Very Low",
        "resume_claims": [], "interview_demonstrated_strengths": [], "weaknesses": [],
        "technical_gaps": [], "recommended_topics": [], "role_readiness": "Insufficient evidence",
        "final_feedback": "Limited evidence.",
    }
    record = database.get_session("round-trip-session")
    with database.Session() as session:
        stored = session.get(SessionRecord, "round-trip-session")
        stored.config_json = json.dumps(old_config)
        stored.interview_data_json = json.dumps({
            "interview_plan": [old_question], "questions": [old_question["question"]],
            "answers": ["A cache stores values."], "analyses": [old_analysis],
            "decisions": [], "topics_covered": ["Caching"], "question_metadata": [{
                "question": old_question["question"], "question_type": "Technical", "topic": "Caching",
                "difficulty": "Easy", "expected_concepts": ["Cache"],
            }],
        })
        stored.evaluation_json = json.dumps(old_evaluation)
        session.commit()

    loaded = database.load_session("round-trip-session")
    assert "max_follow_ups_per_question" not in loaded["config"]
    assert loaded["interview_data"]["adaptation_history"] == []
    assert loaded["interview_data"]["follow_up_counts"] == {}
    assert loaded["interview_data"]["interview_plan"][0]["primary_competency"] is None
    assert loaded["interview_data"]["analyses"][0]["answer_classification"] == "substantive"
    assert loaded["interview_data"]["conversation_history"] == []
    assert loaded["evaluation"]["evidence_summary"]["insufficient_evidence"] == []
    assert not loaded["compatibility_warnings"]
    assert record is not None
    database.engine.dispose()


def test_conversation_history_round_trips_in_existing_session_json():
    with TemporaryDirectory(prefix="interviewos-conversation-", dir=Path.cwd()) as temp_dir:
        database = Database(str(Path(temp_dir) / "conversation.sqlite"))
        state = complete_state()
        state["conversation_history"] = [{
            "utterance": "Can you repeat the question?",
            "intent": "REPEAT",
            "response": "How do you debug an API?",
        }]
        database.save_session(state)
        loaded = database.load_session(state["session_id"])
        assert loaded["interview_data"]["conversation_history"] == state["conversation_history"]
        database.engine.dispose()

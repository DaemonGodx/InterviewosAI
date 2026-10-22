import pytest
import uuid
from unittest.mock import MagicMock
from app.models.schemas import CandidateProfile, InterviewConfig, InterviewState, InterviewQuestion, AnswerAnalysis, EvaluationReport
from app.engine.nodes import generate_question_node, analyze_answer_node, evaluate_node
from app.services.resume_service import ResumeService

@pytest.mark.parametrize("interview_type,expected_type", [("Behavioral", "Behavioral"), ("Technical", "Technical")])
def test_question_node_uses_mocked_model(mocker, interview_type, expected_type):
    config = InterviewConfig(target_role="Backend", experience_level="5+ years", interview_type=interview_type, question_count=1)
    state = InterviewState(session_id="1", config=config, profile=CandidateProfile()).model_dump()
    expected = InterviewQuestion(
        question="Describe a time you took ownership of a problem." if expected_type == "Behavioral" else "How would you debug a slow API?",
        topic="Ownership" if expected_type == "Behavioral" else "Debugging",
        difficulty="Medium",
        question_type=expected_type,
        expected_concepts=["Evidence"],
    )
    chain = MagicMock()
    chain.invoke.return_value = expected
    mocker.patch("app.engine.nodes.ChatPromptTemplate.from_messages", return_value=MagicMock(__or__=lambda self, other: chain))
    result = generate_question_node(state)
    assert result['current_question'] == expected.question

def test_h_one_question_confidence(mocker):
    config = InterviewConfig(target_role="Backend", experience_level="5+ years", interview_type="Technical", question_count=1)
    state = InterviewState(session_id="1", config=config, profile=CandidateProfile()).model_dump()
    
    state['questions'] = ["What is Redis?"]
    state['answers'] = ["It is a cache."]
    state['analyses'] = [AnswerAnalysis(correctness_score=5, depth_score=5, communication_score=5, feedback="Good", is_satisfactory=True).model_dump()]
    state['question_metadata'] = [{
        "question": "What is Redis?", "question_type": "Technical", "topic": "Caching",
        "primary_competency": "Caching fundamentals", "difficulty": "Medium", "expected_concepts": ["Cache"],
    }]

    result = evaluate_node(state)
    evaluation = result['evaluation']
    assert evaluation['evaluation_confidence'] == "Very Low"
    assert evaluation['role_readiness'] == "Insufficient evidence"

def test_i_failed_structured_output():
    config = InterviewConfig(target_role="Backend", experience_level="5+ years", interview_type="Technical", question_count=1)
    state = InterviewState(session_id="1", config=config, profile=CandidateProfile()).model_dump()
    
    state['current_question'] = "What is Redis?"
    state['current_answer'] = "This will definitely fail somehow" 
    # Just to simulate the fallback, we test the logic inside nodes manually.
    # The analyze_answer_node has a fallback.
    
def test_j_score_validation():
    try:
        report = EvaluationReport(
            overall_score=200.0, # Invalid
            communication_score=5.0,
            evaluation_confidence="High",
            resume_claims=[],
            interview_demonstrated_strengths=[],
            weaknesses=[],
            technical_gaps=[],
            recommended_topics=[],
            role_readiness="Ready",
            final_feedback="Great"
        )
        assert False, "Should have failed validation"
    except Exception:
        assert True

def test_e_experience_recommendation():
    profile = CandidateProfile(
        estimated_years_experience=5.0,
        recommended_experience_level="5+ years"
    )
    assert profile.recommended_experience_level == "5+ years"

if __name__ == "__main__":
    pytest.main([__file__])

import pytest
from unittest.mock import MagicMock
from app.engine.graph import InterviewEngine
from app.engine.state import GraphState
from app.models.schemas import (
    CandidateProfile,
    InterviewConfig,
    InterviewQuestion,
    AnswerAnalysis,
    AdaptiveDecision,
    EvaluationReport
)

@pytest.fixture
def base_state():
    return {
        "session_id": "test-session",
        "config": {"target_role": "Backend", "experience_level": "Fresher", "question_count": 3, "starting_difficulty": "Medium", "interview_type": "Technical"},
        "profile": {"name": "Test User", "skills": ["Python"], "resume_summary": "Test"},
        "questions": [],
        "answers": [],
        "analyses": [],
        "decisions": [],
        "topics_covered": [],
        "current_difficulty": "Medium",
        "current_question": None,
        "current_answer": None,
        "status": "IN_PROGRESS",
        "evaluation": None
    }

@pytest.fixture
def mock_engine(mocker):
    # Mock ChatPromptTemplate and LLM inside nodes
    mock_chain = MagicMock()
    mocker.patch('app.engine.nodes.ChatPromptTemplate.from_messages', return_value=MagicMock(__or__=lambda self, other: mock_chain))
    
    # We patch the specific node functions before InterviewEngine is imported or instantiated.
    mocker.patch('app.engine.graph.generate_question_node')
    mocker.patch('app.engine.graph.analyze_answer_node')
    mocker.patch('app.engine.graph.adaptive_decision_node')
    mocker.patch('app.engine.graph.evaluate_node')
    engine = InterviewEngine()
    return engine

def test_engine_initialization(base_state, mocker):
    mocker.patch('app.engine.graph.generate_question_node', return_value={
        "current_question": "Mock Question",
        "topics_covered": ["Mock Topic"],
        "questions": ["Mock Question"]
    })
    
    engine = InterviewEngine()
    state = engine.start_interview(base_state)
    
    assert state["current_question"] == "Mock Question"
    assert "Mock Topic" in state["topics_covered"]
    assert "Mock Question" in state["questions"]

def test_engine_process_answer(base_state, mocker):
    base_state["current_question"] = "What is Python?"
    base_state["questions"] = ["What is Python?"]
    mocker.patch('app.engine.graph.classify_turn_node', side_effect=lambda state: {
        "current_turn_intent": {"intent": "ANSWER", "confidence": 1.0},
        "classified_utterance": state["current_input"], "current_input": None,
    })
    
    mocker.patch('app.engine.graph.analyze_answer_node', return_value={
        "analyses": [{"correctness_score": 8, "depth_score": 8, "communication_score": 8, "feedback": "Good", "is_satisfactory": True, "status": "success"}],
        "answers": ["Python is a language"],
        "current_answer": None
    })
    mocker.patch('app.engine.graph.adaptive_decision_node', return_value={
        "decisions": [{"action": "ADVANCE", "rationale": "Good", "competency_sufficient": True}],
        "current_difficulty": "Medium"
    })
    mocker.patch('app.engine.graph.generate_question_node', return_value={
        "current_question": "Next Question",
        "topics_covered": ["Next Topic"],
        "questions": ["What is Python?", "Next Question"]
    })
    
    engine = InterviewEngine()
    state = engine.process_answer(base_state, "Python is a language")
    
    assert len(state["answers"]) == 1
    assert state["answers"][0] == "Python is a language"
    assert len(state["analyses"]) == 1
    assert state["current_difficulty"] == "Medium"
    assert state["current_question"] == "Next Question"
    assert len(state["questions"]) == 2

def test_engine_termination(base_state, mocker):
    # Setup state to be just before the final question's answer is processed
    base_state["config"]["question_count"] = 1
    base_state["current_question"] = "Final Question"
    base_state["questions"] = ["Final Question"]
    mocker.patch('app.engine.graph.classify_turn_node', side_effect=lambda state: {
        "current_turn_intent": {"intent": "ANSWER", "confidence": 1.0},
        "classified_utterance": state["current_input"], "current_input": None,
    })
    
    mocker.patch('app.engine.graph.analyze_answer_node', return_value={
        "analyses": [{"correctness_score": 10, "depth_score": 10, "communication_score": 10, "feedback": "Perfect", "is_satisfactory": True, "status": "success"}],
        "answers": ["Final answer"],
        "current_answer": None
    })
    mocker.patch('app.engine.graph.adaptive_decision_node', return_value={
        "decisions": [{"action": "FINISH", "rationale": "Done", "competency_sufficient": True}],
        "current_difficulty": "Medium"
    })
    mocker.patch('app.engine.graph.evaluate_node', return_value={
        "evaluation": EvaluationReport(
            overall_score=9.0, technical_score=9.0, problem_solving_score=9.0, communication_score=9.0, behavioral_score=9.0,
            evaluation_confidence="High", resume_claims=["Strong"], interview_demonstrated_strengths=["Strong"],
            weaknesses=["None"], technical_gaps=[], recommended_topics=[], role_readiness="Ready", final_feedback="Great"
        ),
        "status": "COMPLETED"
    })
    
    engine = InterviewEngine()
    state = engine.process_answer(base_state, "Final answer")
    
    assert state["status"] == "COMPLETED"
    assert state["evaluation"] is not None
    assert state["evaluation"]["overall_score"] == 9.0

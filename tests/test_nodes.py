import pytest
from unittest.mock import MagicMock
from app.engine.nodes import analyze_answer_node, generate_question_node
from app.models.schemas import AnswerAnalysis, InterviewQuestion

@pytest.fixture
def base_state():
    return {
        "session_id": "test",
        "config": {"interview_type": "Mixed", "question_count": 5, "target_role": "Backend", "experience_level": "Mid"},
        "profile": {"resume_summary": "Test resume"},
        "questions": [],
        "topics_covered": [],
        "current_difficulty": "Medium"
    }

def mock_llm_chain(mocker, expected_returns):
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = expected_returns
    # The nodes.py uses: chain = prompt | structured_llm
    # We can patch 'app.engine.nodes.llm.with_structured_output' and 'app.engine.nodes.ChatPromptTemplate.from_messages'
    # Actually, simpler: patch the pipeline.
    # In nodes.py: chain = prompt | structured_llm
    # We can patch ChatPromptTemplate.from_messages to return a mock prompt, and then mock the __or__ method to return mock_chain.
    mocker.patch('app.engine.nodes.ChatPromptTemplate.from_messages', return_value=MagicMock(__or__=lambda self, other: mock_chain))
    return mock_chain

def test_mixed_mode_alternating_5_questions(base_state, mocker):
    base_state["config"]["interview_type"] = "Mixed"
    base_state["config"]["question_count"] = 5
    
    # Q1: expected Technical
    mock_llm_chain(mocker, [InterviewQuestion(question="Q1", topic="T1", difficulty="Medium", question_type="Technical", expected_concepts=["C1"])])
    res1 = generate_question_node(base_state)
    assert res1["current_question_type"] == "Technical"
    base_state.update(res1)
    
    # Q2: expected Behavioral
    mock_llm_chain(mocker, [InterviewQuestion(question="Q2", topic="T2", difficulty="Medium", question_type="Behavioral", expected_concepts=["C2"])])
    res2 = generate_question_node(base_state)
    assert res2["current_question_type"] == "Behavioral"
    base_state.update(res2)
    
    # Q3: expected Technical
    mock_llm_chain(mocker, [InterviewQuestion(question="Q3", topic="T3", difficulty="Medium", question_type="Technical", expected_concepts=["C3"])])
    res3 = generate_question_node(base_state)
    assert res3["current_question_type"] == "Technical"
    base_state.update(res3)

    # Q4: expected Behavioral
    mock_llm_chain(mocker, [InterviewQuestion(question="Q4", topic="T4", difficulty="Medium", question_type="Behavioral", expected_concepts=["C4"])])
    res4 = generate_question_node(base_state)
    assert res4["current_question_type"] == "Behavioral"
    base_state.update(res4)

    # Q5: expected Technical
    mock_llm_chain(mocker, [InterviewQuestion(question="Q5", topic="T5", difficulty="Medium", question_type="Technical", expected_concepts=["C5"])])
    res5 = generate_question_node(base_state)
    assert res5["current_question_type"] == "Technical"

def test_technical_mode_only(base_state, mocker):
    base_state["config"]["interview_type"] = "Technical"
    
    mock_llm_chain(mocker, [InterviewQuestion(question="Q1", topic="T1", difficulty="Medium", question_type="Technical", expected_concepts=["C1"])])
    res1 = generate_question_node(base_state)
    assert res1["current_question_type"] == "Technical"
    base_state.update(res1)
    
    mock_llm_chain(mocker, [InterviewQuestion(question="Q2", topic="T2", difficulty="Medium", question_type="Technical", expected_concepts=["C2"])])
    res2 = generate_question_node(base_state)
    assert res2["current_question_type"] == "Technical"

def test_behavioral_mode_only(base_state, mocker):
    base_state["config"]["interview_type"] = "Behavioral"
    
    mock_llm_chain(mocker, [InterviewQuestion(question="Q1", topic="T1", difficulty="Medium", question_type="Behavioral", expected_concepts=["C1"])])
    res1 = generate_question_node(base_state)
    assert res1["current_question_type"] == "Behavioral"
    base_state.update(res1)
    
    mock_llm_chain(mocker, [InterviewQuestion(question="Q2", topic="T2", difficulty="Medium", question_type="Behavioral", expected_concepts=["C2"])])
    res2 = generate_question_node(base_state)
    assert res2["current_question_type"] == "Behavioral"

def test_metadata_preservation(base_state, mocker):
    mock_llm_chain(mocker, [InterviewQuestion(
        question="What is Redis?",
        topic="Caching",
        difficulty="Medium",
        question_type="Technical",
        expected_concepts=["Cache invalidation"]
    )])
    
    res = generate_question_node(base_state)
    
    assert res["current_question"] == "What is Redis?"
    assert res["current_question_type"] == "Technical"
    assert res["current_question_topic"] == "Caching"
    assert res["current_question_difficulty"] == "Medium"
    assert "Cache invalidation" in res["current_expected_concepts"]
    
    assert len(res["question_metadata"]) == 1
    meta = res["question_metadata"][0]
    assert meta["question"] == "What is Redis?"
    assert meta["question_type"] == "Technical"
    assert meta["topic"] == "Caching"
    assert meta["difficulty"] == "Medium"
    assert "Cache invalidation" in meta["expected_concepts"]

def test_generate_question_uses_next_confirmed_plan_item(base_state, mocker):
    base_state["interview_plan"] = [
        {
            "question": "Planned question one",
            "topic": "Caching",
            "difficulty": "Hard",
            "question_type": "Technical",
            "expected_concepts": ["Invalidation", "Consistency"],
        },
        {
            "question": "Planned question two",
            "topic": "Teamwork",
        "primary_competency": "Conflict resolution",
            "difficulty": "Hard",
            "question_type": "Behavioral",
            "expected_concepts": ["Ownership"],
        },
    ]
    base_state["questions"] = ["Already asked"]
    base_state["plan_cursor"] = 1

    result = generate_question_node(base_state)

    assert result["current_question"] == "Planned question two"
    assert result["current_question_type"] == "Behavioral"
    assert result["current_question_topic"] == "Teamwork"
    assert result["current_primary_competency"] == "Conflict resolution"
    assert result["current_difficulty"] == "Hard"
    assert result["current_expected_concepts"] == ["Ownership"]
    assert result["questions"] == ["Already asked", "Planned question two"]

def test_mismatched_output_retries(base_state, mocker):
    base_state["config"]["interview_type"] = "Technical"
    
    # First returns Behavioral (mismatch), then Technical (match)
    mock_chain = mock_llm_chain(mocker, [
        InterviewQuestion(question="B1", topic="T1", difficulty="Medium", question_type="Behavioral", expected_concepts=[]),
        InterviewQuestion(question="T1", topic="T2", difficulty="Medium", question_type="Technical", expected_concepts=[])
    ])
    
    res = generate_question_node(base_state)
    assert mock_chain.invoke.call_count == 2
    assert res["current_question_type"] == "Technical"
    assert res["current_question"] == "T1"

def test_fallback_respects_expected_type(base_state, mocker):
    # Force all 3 attempts to fail
    mock_chain = mock_llm_chain(mocker, [Exception("Error1"), Exception("Error2"), Exception("Error3")])
    
    base_state["config"]["interview_type"] = "Technical"
    res1 = generate_question_node(base_state)
    assert res1["current_question_type"] == "Technical"
    assert "debugging" in res1["current_question"].lower()
    
    base_state["config"]["interview_type"] = "Behavioral"
    res2 = generate_question_node(base_state)
    assert res2["current_question_type"] == "Behavioral"
    assert "difficult situation" in res2["current_question"].lower()


def analysis_state(question_type):
    return {
        "config": {"target_role": "Backend Engineer", "experience_level": "Mid-level"},
        "current_question": "Describe your approach to the problem.",
        "current_question_type": question_type,
        "current_question_topic": "Incident response" if question_type == "Technical" else "Teamwork",
        "current_primary_competency": "Production debugging" if question_type == "Technical" else "Conflict resolution",
        "current_question_difficulty": "Hard",
        "current_expected_concepts": ["Diagnostics", "Trade-offs"] if question_type == "Technical" else ["Listening", "Resolution"],
        "current_difficulty": "Easy",  # analysis must use the current question's own difficulty
        "current_answer": "I investigated the issue and communicated the resolution.",
        "answers": [],
        "analyses": [],
    }


@pytest.mark.parametrize("question_type,expected_rubric", [
    ("Technical", "TECHNICAL RUBRIC"),
    ("Behavioral", "BEHAVIORAL RUBRIC"),
])
def test_analysis_invocation_includes_question_metadata_and_matching_rubric(mocker, question_type, expected_rubric):
    chain = MagicMock()
    chain.invoke.return_value = AnswerAnalysis(
        correctness_score=7, depth_score=6, communication_score=8,
        feedback="Clear evidence and a useful next step.", is_satisfactory=True,
    )
    mocker.patch("app.engine.nodes.ChatPromptTemplate.from_messages", return_value=MagicMock(__or__=lambda self, other: chain))
    llm_mock = MagicMock()
    mocker.patch("app.engine.nodes.llm", llm_mock)

    result = analyze_answer_node(analysis_state(question_type))

    call = chain.invoke.call_args.args[0]
    assert call["question_type"] == question_type
    assert call["topic"] == ("Incident response" if question_type == "Technical" else "Teamwork")
    assert call["primary_competency"] == ("Production debugging" if question_type == "Technical" else "Conflict resolution")
    assert call["difficulty"] == "Hard"
    assert call["expected_concepts"] in {"Diagnostics, Trade-offs", "Listening, Resolution"}
    assert call["role"] == "Backend Engineer"
    assert call["experience"] == "Mid-level"
    assert expected_rubric in call["rubric"]
    assert result["analyses"][0]["status"] == "success"


def test_blank_answer_is_classified_without_calling_analysis_model(mocker):
    chain = MagicMock()
    mocker.patch("app.engine.nodes.ChatPromptTemplate.from_messages", return_value=MagicMock(__or__=lambda self, other: chain))
    llm_mock = MagicMock()
    mocker.patch("app.engine.nodes.llm", llm_mock)
    state = analysis_state("Behavioral")
    state["current_answer"] = "  \n"

    result = analyze_answer_node(state)

    assert result["analyses"][0]["answer_classification"] == "blank"
    assert result["analyses"][0]["status"] == "success"
    assert result["analyses"][0]["is_satisfactory"] is False
    chain.invoke.assert_not_called()


def test_invalid_analysis_output_retries_and_is_marked_failed(mocker):
    chain = MagicMock()
    chain.invoke.return_value = {"correctness_score": 11, "depth_score": 4, "communication_score": 4, "feedback": "bad", "is_satisfactory": True}
    mocker.patch("app.engine.nodes.ChatPromptTemplate.from_messages", return_value=MagicMock(__or__=lambda self, other: chain))
    llm_mock = MagicMock()
    mocker.patch("app.engine.nodes.llm", llm_mock)

    result = analyze_answer_node(analysis_state("Technical"))

    assert chain.invoke.call_count == 3
    assert result["analyses"][0]["status"] == "failed"
    assert result["analyses"][0]["answer_classification"] == "analysis_failed"
    assert "No candidate score was assigned" in result["analyses"][0]["feedback"]


def test_analysis_service_failure_is_bounded_and_never_becomes_candidate_score(mocker):
    chain = MagicMock()
    chain.invoke.side_effect = RuntimeError("analysis service unavailable")
    mocker.patch("app.engine.nodes.ChatPromptTemplate.from_messages", return_value=MagicMock(__or__=lambda self, other: chain))
    llm_mock = MagicMock()
    mocker.patch("app.engine.nodes.llm", llm_mock)

    result = analyze_answer_node(analysis_state("Behavioral"))

    assert chain.invoke.call_count == 3
    analysis = result["analyses"][0]
    assert analysis["status"] == "failed"
    assert analysis["is_satisfactory"] is False
    assert analysis["answer_classification"] == "analysis_failed"

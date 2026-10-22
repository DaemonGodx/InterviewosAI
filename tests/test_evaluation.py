import pytest

from app.engine.nodes import StateConsistencyError, evaluate_node, validate_evaluation_state
from app.models.schemas import AnswerAnalysis


def evaluation_state(categories=None, scores=None, classifications=None, statuses=None, count=None):
    categories = categories or ["Technical", "Behavioral", "Technical"]
    scores = scores or [(8, 7, 8), (7, 6, 9), (4, 5, 6)]
    classifications = classifications or ["substantive"] * len(categories)
    statuses = statuses or ["success"] * len(categories)
    plan = []
    questions = []
    answers = []
    analyses = []
    metadata = []
    for index, category in enumerate(categories):
        question = f"Question {index + 1}"
        topic = f"Competency {index + 1}"
        questions.append(question)
        answers.append(f"Answer {index + 1}")
        correctness, depth, communication = scores[index]
        analyses.append(AnswerAnalysis(
            correctness_score=correctness,
            depth_score=depth,
            communication_score=communication,
            feedback="Evidence-based feedback.",
            is_satisfactory=correctness >= 5,
            status=statuses[index],
            answer_classification=classifications[index],
        ).model_dump())
        metadata.append({
            "question": question, "question_type": category, "topic": topic,
            "primary_competency": topic, "difficulty": "Medium", "expected_concepts": ["Evidence"],
            "plan_index": index, "is_follow_up": False,
        })
        plan.append({
            "question": question, "question_type": category, "topic": topic,
            "primary_competency": topic, "difficulty": "Medium", "expected_concepts": ["Evidence"],
        })
    configured = count if count is not None else len(categories)
    return {
        "config": {"question_count": configured, "interview_type": "Mixed", "target_role": "Engineer"},
        "profile": {"inferred_strengths": ["resume claim"]},
        "interview_plan": plan,
        "plan_cursor": len(questions),
        "plan_execution_order": list(range(configured)),
        "questions": questions, "answers": answers, "analyses": analyses,
        "question_metadata": metadata,
    }


@pytest.mark.parametrize("field", ["answers", "analyses", "question_metadata"])
def test_evaluation_detects_parallel_history_length_mismatch(field):
    state = evaluation_state()
    state[field].pop()

    with pytest.raises(StateConsistencyError, match="lengths do not align"):
        validate_evaluation_state(state)

    result = evaluate_node(state)
    assert result["status"] == "COMPLETED_WITH_ERRORS"
    assert "evaluation" not in result
    assert "lengths do not align" in result["evaluation_error"]


def test_evaluation_rejects_question_metadata_that_does_not_match_question():
    state = evaluation_state()
    state["question_metadata"][0]["question"] = "A different question"
    with pytest.raises(StateConsistencyError, match="does not match the asked question"):
        validate_evaluation_state(state)


def test_failed_analysis_is_unavailable_evidence_not_a_zero_candidate_score():
    state = evaluation_state(categories=["Technical"], scores=[(0, 0, 0)], statuses=["failed"], classifications=["analysis_failed"])

    report = evaluate_node(state)["evaluation"]

    assert report["overall_score"] is None
    assert report["technical_score"] is None
    assert report["behavioral_score"] is None
    assert report["evaluation_confidence"] == "Very Low"
    assert report["role_readiness"] == "Insufficient evidence"
    assert report["evidence_summary"]["insufficient_evidence"]
    assert not report["evidence_summary"]["demonstrated_knowledge_gaps"]
    assert "analysis failed" in report["final_feedback"]


def test_successful_blank_analyses_are_not_misreported_as_analysis_failures():
    state = evaluation_state(
        categories=["Behavioral", "Behavioral"],
        scores=[(0, 0, 0), (1, 0, 1)],
        statuses=["success", "success"],
        classifications=["blank", "blank"],
    )

    report = evaluate_node(state)["evaluation"]

    assert report["overall_score"] is None
    assert report["behavioral_score"] is None
    assert report["role_readiness"] == "Insufficient evidence"
    assert "2 response(s) were blank" in report["final_feedback"]
    assert "analysis failed" not in report["final_feedback"]
    assert "No usable answer analyses were available" not in report["final_feedback"]
    assert all("no answer was provided" in item for item in report["evidence_summary"]["insufficient_evidence"])


def test_successful_low_score_is_still_scored_when_analysis_has_usable_evidence():
    state = evaluation_state(
        categories=["Technical"], scores=[(0, 0, 0)],
        statuses=["success"], classifications=["partially_correct"],
    )

    report = evaluate_node(state)["evaluation"]

    assert report["overall_score"] == 0
    assert report["technical_score"] == 0
    assert report["evaluation_confidence"] == "Very Low"
    assert report["technical_gaps"]


def test_overall_score_matches_documented_weighted_formula_for_observed_run():
    state = evaluation_state(
        categories=["Technical", "Behavioral", "Behavioral"],
        scores=[(0, 0, 2), (1, 1, 2), (0, 0, 3)],
        statuses=["success"] * 3,
        classifications=["admitted_unknown", "partially_correct", "admitted_unknown"],
    )

    report = evaluate_node(state)["evaluation"]

    assert report["overall_score"] == 7.33
    assert report["technical_score"] == 0
    assert report["problem_solving_score"] == 0
    assert report["behavioral_score"] == 0.9
    assert report["communication_score"] == 2.33


@pytest.mark.parametrize("count", [1, 2])
def test_one_or_two_usable_answers_cannot_support_readiness_claims(count):
    categories = ["Technical", "Behavioral"][:count]
    state = evaluation_state(categories=categories, scores=[(10, 10, 10)] * count)

    report = evaluate_node(state)["evaluation"]

    assert report["evaluation_confidence"] == "Very Low"
    assert report["role_readiness"] == "Insufficient evidence"
    assert "cannot establish overall role readiness" in report["readiness_basis"]


def test_category_scores_are_available_only_for_assessed_categories():
    technical = evaluation_state(categories=["Technical"], scores=[(8, 6, 9)])
    technical_report = evaluate_node(technical)["evaluation"]
    assert technical_report["technical_score"] == 8
    assert technical_report["problem_solving_score"] == 7.2
    assert technical_report["behavioral_score"] is None

    behavioral = evaluation_state(categories=["Behavioral"], scores=[(8, 6, 9)])
    behavioral_report = evaluate_node(behavioral)["evaluation"]
    assert behavioral_report["technical_score"] is None
    assert behavioral_report["problem_solving_score"] is None
    assert behavioral_report["behavioral_score"] == 7.6


def test_score_aggregation_is_transparent_bounded_and_excludes_failed_analyses():
    state = evaluation_state(
        categories=["Technical", "Technical"],
        scores=[(8, 6, 9), (0, 0, 0)],
        statuses=["success", "failed"],
        classifications=["substantive", "analysis_failed"],
    )
    report = evaluate_node(state)["evaluation"]

    assert report["overall_score"] == 76.0
    assert report["technical_score"] == 8
    assert report["problem_solving_score"] == 7.2
    assert report["communication_score"] == 9
    assert 0 <= report["overall_score"] <= 100
    for name in ("technical_score", "problem_solving_score", "communication_score", "behavioral_score"):
        value = report[name]
        assert value is None or 0 <= value <= 10


def test_confidence_uses_usable_answer_count_and_competency_breadth():
    state = evaluation_state(categories=["Technical"] * 6, scores=[(8, 8, 8)] * 6)
    for index in range(6):
        state["question_metadata"][index]["primary_competency"] = f"Distinct competency {index}"
        state["question_metadata"][index]["topic"] = f"Distinct competency {index}"
    report = evaluate_node(state)["evaluation"]
    assert report["evaluation_confidence"] == "High"

    narrow = evaluation_state(categories=["Technical"] * 6, scores=[(8, 8, 8)] * 6)
    for metadata in narrow["question_metadata"]:
        metadata["primary_competency"] = "One competency"
        metadata["topic"] = "One competency"
    assert evaluate_node(narrow)["evaluation"]["evaluation_confidence"] == "Low"


def test_assessment_distinguishes_demonstration_partial_evidence_and_knowledge_gap():
    state = evaluation_state(
        categories=["Technical", "Behavioral", "Technical"],
        scores=[(9, 8, 8), (5, 5, 7), (2, 3, 4)],
    )
    summary = evaluate_node(state)["evaluation"]["evidence_summary"]
    assert summary["demonstrated_competence"]
    assert summary["partial_evidence"]
    assert summary["demonstrated_knowledge_gaps"]
    assert "general suitability" in evaluate_node(state)["evaluation"]["final_feedback"].lower()


from io import StringIO
from contextlib import nullcontext
from types import SimpleNamespace

from rich.console import Console

import app.__main__ as cli
from app.models.schemas import (
    CandidateProfile, InterviewConfig, InterviewPlan, InterviewQuestion,
    QuestionSemanticValidation,
)


def test_display_evaluation_prints_model_text_without_markup(mocker):
    output = StringIO()
    mocker.patch.object(cli, "console", Console(file=output, force_terminal=False, color_system=None))

    cli.display_evaluation({
        "overall_score": 0,
        "evaluation_confidence": "Very Low",
        "communication_score": 0,
        "behavioral_score": 0,
        "resume_claims": ["Claim [custom] text"],
        "interview_demonstrated_strengths": [],
        "weaknesses": ["No answer [tag]"],
        "role_readiness": "Insufficient evidence",
        "final_feedback": "Literal [bold]feedback[/not-a-rich-tag]",
    })

    rendered = output.getvalue()
    assert "INTERVIEWOS REPORT" in rendered
    assert "Literal [bold]feedback[/not-a-rich-tag]" in rendered
    assert "No answer [tag]" in rendered
    assert "mean across usable answers" in rendered
    assert "50% correctness/relevance + 30% depth + 20% communication" in " ".join(rendered.split())


def test_answer_feedback_does_not_show_a_candidate_score_for_blank_or_failed_analysis(mocker):
    output = StringIO()
    mocker.patch.object(cli, "console", Console(file=output, force_terminal=False, color_system=None))

    cli.display_answer_analysis({
        "status": "success", "answer_classification": "blank",
        "correctness_score": 0, "feedback": "No answer was provided.",
    })
    cli.display_answer_analysis({
        "status": "failed", "answer_classification": "analysis_failed",
        "correctness_score": 0, "feedback": "Analysis could not be completed.",
    })

    rendered = output.getvalue()
    assert "Analysis Score:" not in rendered
    assert "No answer evidence; no candidate score assigned" in rendered
    assert "Analysis unavailable; no candidate score assigned" in rendered


def test_plan_confirmation_edit_and_single_question_regeneration(mocker):
    intended = QuestionSemanticValidation(
        matches_intended_category=True,
        primary_evaluation_objective="debugging approach",
        reason="The question assesses engineering problem solving.",
    )
    original = InterviewQuestion(
        question="How would you debug a slow API?", topic="Debugging",
        primary_competency="Troubleshooting", difficulty="Medium",
        question_type="Technical", expected_concepts=["Logs", "Metrics"],
    )
    edited = original.model_copy(update={"question": "How would you investigate API latency?"})
    regenerated = original.model_copy(update={"question": "How do you isolate a production timeout?"})
    plan = InterviewPlan(questions=[original], semantic_validations=[intended])

    class FakePlanner:
        def generate_plan(self, _profile, _config, progress_callback=None):
            return plan

        def validate_plan(self, _plan, _config):
            return None

        def edit_question(self, _plan, _config, index, text):
            assert index == 0 and text == edited.question
            plan.questions[0] = edited
            return intended

        def regenerate_question(self, _plan, _profile, _config, index):
            assert index == 0
            plan.questions[0] = regenerated
            return SimpleNamespace(repair_warning=None)

    mocker.patch.object(cli, "PlannerService", FakePlanner)
    mocker.patch.object(cli, "review_interview_plan")
    mocker.patch.object(cli.console, "status", return_value=nullcontext())
    config = InterviewConfig(target_role="Backend", experience_level="Junior", question_count=1)

    mocker.patch.object(cli.Prompt, "ask", side_effect=["2", "How would you investigate API latency?", "3", "1", "1"])
    mocker.patch.object(cli.IntPrompt, "ask", return_value=1)
    result = cli.prepare_interview_plan(CandidateProfile(), config)

    assert result[0].question == regenerated.question
    assert result[0].question_type == "Technical"

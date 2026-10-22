from unittest.mock import MagicMock

import pytest

from app.models.schemas import (
    CandidateProfile,
    InterviewConfig,
    InterviewPlan,
    InterviewQuestion,
    QuestionSemanticValidation,
    GeneratedQuestionReview,
)
from app.services.planner_service import PlannerService, QuestionSemanticError


def question(text="A technical question", category="Technical", difficulty="Easy", behavioral_category=None):
    return InterviewQuestion(
        question=text,
        topic="Core skills",
        difficulty=difficulty,
        question_type=category,
        behavioral_category=behavioral_category or ("HR_MOTIVATION" if category == "Behavioral" else None),
        expected_concepts=["Reasoning", "Evidence"],
    )


def validation(matches=True, objective="Technical reasoning", reason="The expected answer demonstrates technical reasoning.", duplicate=False, assessed=None):
    return QuestionSemanticValidation(
        matches_intended_category=matches,
        matches_intended_competency=True,
        matches_intended_behavioral_category=True,
        primary_evaluation_objective=objective,
        reason=reason,
        assessed_competency=assessed or objective,
        is_semantic_duplicate=duplicate,
        duplicate_reason="Same underlying competency" if duplicate else None,
    )


def configured_service(mocker, question_results=(), validation_results=(), max_attempts=3):
    llm = MagicMock()
    question_model = MagicMock()
    semantic_model = MagicMock()
    combined_model = MagicMock()
    question_model.invoke.side_effect = list(question_results)
    semantic_model.invoke.side_effect = list(validation_results)

    def combined_invoke(invocation):
        generated = question_model.invoke(invocation)
        reviewed = semantic_model.invoke(invocation)
        return GeneratedQuestionReview.model_validate({
            "question": generated, "semantic_validation": reviewed,
        })

    combined_model.invoke.side_effect = combined_invoke
    llm.with_structured_output.side_effect = lambda schema: (
        combined_model if schema is GeneratedQuestionReview
        else question_model if schema is InterviewQuestion
        else semantic_model
    )
    mocker.patch(
        "app.services.planner_service.ChatPromptTemplate.from_messages",
        return_value=MagicMock(__or__=lambda self, model: model),
    )
    return PlannerService(llm=llm, max_attempts=max_attempts), question_model, semantic_model


def config(kind="Technical", count=1, difficulty="Hard"):
    return InterviewConfig(
        target_role="Backend Developer",
        experience_level="Mid-level",
        interview_type=kind,
        question_count=count,
        starting_difficulty=difficulty,
    )


@pytest.mark.parametrize("count,technical,behavioral", [(2, 1, 1), (5, 3, 2), (10, 6, 4)])
def test_mixed_plan_distribution_for_common_counts(mocker, count, technical, behavioral):
    schedule = PlannerService.expected_types("Mixed", count)
    behavioral_schedule = PlannerService.behavioral_categories_for_slots(schedule)
    generated = [
        question(f"Question {i}", category=category, difficulty="Hard", behavioral_category=behavioral_schedule[i])
        for i, category in enumerate(schedule)
    ]
    validations = [validation(True, f"{category} objective", f"The objective is {category.lower()}.") for category in schedule]
    service, _, _ = configured_service(mocker, generated, validations)

    plan = service.generate_plan(CandidateProfile(), config("Mixed", count))

    categories = [item.question_type for item in plan.questions]
    assert categories.count("Technical") == technical
    assert categories.count("Behavioral") == behavioral
    assert set(categories) == {"Technical", "Behavioral"}


def test_plan_generation_uses_one_model_round_trip_per_valid_question_and_reports_progress(mocker):
    count = 3
    schedule = PlannerService.expected_types("Technical", count)
    generated = [question(f"Question {i}", category="Technical", difficulty="Hard") for i in range(count)]
    validations = [validation(True, f"Objective {i}", "The objective matches the assigned technical competency.") for i in range(count)]
    service, question_model, semantic_model = configured_service(mocker, generated, validations)
    progress = []

    plan = service.generate_plan(
        CandidateProfile(), config("Technical", count),
        progress_callback=lambda *args: progress.append(args),
    )

    assert len(plan.questions) == count
    assert service.llm.with_structured_output.call_count == count
    assert question_model.invoke.call_count == semantic_model.invoke.call_count == count
    assert [item[0] for item in progress] == [1, 2, 3]
    assert all(item[1] == count for item in progress)


def test_one_question_mixed_is_deterministically_technical():
    assert PlannerService.expected_types("Mixed", 1) == ["Technical"]


def test_technical_question_with_workplace_story_framing_stays_technical(mocker):
    story_question = question(
        "Tell me about a time you redesigned a service. Explain the architecture and the trade-offs you made.",
        category="Technical",
    )
    service, question_model, semantic_model = configured_service(
        mocker,
        [story_question],
        [validation(True, "Architecture and engineering trade-offs", "The answer is judged primarily on technical design choices.")],
    )

    plan = service.generate_plan(CandidateProfile(), config())

    assert plan.questions[0].question == story_question.question
    assert plan.questions[0].question_type == "Technical"
    assert plan.semantic_validations[0].matches_intended_category
    question_model.invoke.assert_called_once()
    semantic_model.invoke.assert_called_once()


def test_microservice_design_mislabeled_behavioral_is_repaired(mocker):
    mislabeled = question(
        "Tell me about a time you designed a microservice. Explain its architecture, caching, and scaling trade-offs.",
        category="Behavioral",
    )
    genuine_behavioral = question(
        "What interests you about this backend developer role?",
        category="Behavioral",
    )
    service, question_model, semantic_model = configured_service(
        mocker,
        [mislabeled, genuine_behavioral],
        [
            validation(False, "System architecture", "The answer is primarily assessed on microservice design and technical trade-offs."),
            validation(True, "Role motivation", "The answer is primarily assessed on role interest and motivation."),
        ],
    )

    plan = service.generate_plan(CandidateProfile(), config("Behavioral"))

    assert plan.questions[0].question == genuine_behavioral.question
    assert plan.questions[0].question_type == "Behavioral"
    assert question_model.invoke.call_count == 2
    assert semantic_model.invoke.call_count == 2


@pytest.mark.parametrize("text,objective", [
    ("Tell me about yourself and what led you to this role.", "Career narrative and motivation"),
    ("Why are you interested in this backend developer role?", "Role motivation"),
    ("What are your strengths and what would you like to improve?", "Self-awareness"),
    ("What would you like to learn or accomplish over the next few years?", "Career goals"),
])
def test_hr_motivation_examples_pass_semantic_validation(mocker, text, objective):
    service, _, _ = configured_service(
        mocker,
        [question(text, category="Behavioral")],
        [validation(True, objective, "The expected evidence is motivation or self-awareness.")],
    )

    plan = service.generate_plan(CandidateProfile(), config("Behavioral"))

    assert plan.questions[0].question == text
    assert plan.semantic_validations[0].matches_intended_category


def test_generation_and_validation_retries_are_bounded_then_use_safe_fallback(mocker):
    invalid = question("Tell me about a time you built a distributed cache.", category="Behavioral")
    service, question_model, semantic_model = configured_service(
        mocker,
        [invalid, invalid],
        [
            validation(False, "Technical implementation", "The primary objective is technical implementation."),
            validation(False, "Technical implementation", "The primary objective is technical implementation."),
        ],
        max_attempts=2,
    )

    plan = service.generate_plan(CandidateProfile(), config("Behavioral"))

    assert question_model.invoke.call_count == 2
    assert semantic_model.invoke.call_count == 2
    assert plan.questions[0].question_type == "Behavioral"
    assert "Tell me about" in plan.questions[0].question
    assert plan.questions[0].topic == "Professional introduction and career narrative"
    assert plan.semantic_validations[0].assessed_competency == "Professional introduction and career narrative"
    assert plan.questions[0].difficulty == "Hard"
    assert plan.questions[0].expected_concepts
    assert plan.repair_warnings and "safe fallback" in plan.repair_warnings[0]


@pytest.mark.parametrize("question_result,validation_result", [
    (RuntimeError("provider unavailable"), None),
    ({"question": "missing fields"}, None),
    (question("A question", category="Technical"), RuntimeError("validator unavailable")),
    (question("A question", category="Technical"), {"not_matches": True}),
])
def test_model_errors_and_malformed_structured_results_fall_back(mocker, question_result, validation_result):
    question_results = [question_result, question_result]
    validation_results = [validation_result, validation_result] if validation_result is not None else []
    service, _, _ = configured_service(mocker, question_results, validation_results, max_attempts=2)

    plan = service.generate_plan(CandidateProfile(), config())

    assert plan.questions[0].question_type == "Technical"
    assert plan.questions[0].difficulty == "Hard"
    assert plan.repair_warnings
    assert plan.semantic_validations[0].matches_intended_category


def test_edit_revalidates_category_and_preserves_old_question_when_rejected(mocker):
    original = question("How would you debug a slow API?", category="Technical", difficulty="Hard")
    original_validation = validation(True, "Debugging", "The answer is judged on a technical approach.")
    service, _, semantic_model = configured_service(
        mocker,
        [],
        [
            validation(False, "Architecture", "This edit asks for system design."),
            validation(True, "Debugging", "The edit evaluates technical troubleshooting."),
        ],
    )
    plan = InterviewPlan(questions=[original], semantic_validations=[original_validation])

    with pytest.raises(QuestionSemanticError):
        service.edit_question(plan, config(), 0, "Tell me about a time you designed a distributed system.")
    assert plan.questions[0] is original

    service.edit_question(plan, config(), 0, "How would you investigate a production latency regression?")
    assert plan.questions[0].question == "How would you investigate a production latency regression?"
    assert plan.questions[0].question_type == "Technical"
    assert semantic_model.invoke.call_count == 2


def test_regenerating_one_question_preserves_other_questions(mocker):
    first = question("How would you debug an API?", category="Technical", difficulty="Hard")
    second = question("Describe a team disagreement.", category="Behavioral", difficulty="Hard")
    service, question_model, _ = configured_service(
        mocker,
        [question("What interests you about this backend developer role?", category="Behavioral")],
        [validation(True, "Role motivation", "The answer focuses on role interest and motivation.")],
    )
    plan = InterviewPlan(
        questions=[first, second],
        semantic_validations=[
            validation(True, "Technical debugging", "The answer evaluates a technical approach."),
            validation(True, "Teamwork", "The answer evaluates professional behavior."),
        ],
    )

    result = service.regenerate_question(plan, CandidateProfile(), config("Mixed", 2), 1)

    assert plan.questions[0] is first
    assert plan.questions[1] is result.question
    assert plan.questions[1].question_type == "Behavioral"
    assert plan.questions[1].difficulty == "Hard"
    generation_input = question_model.invoke.call_args.args[0]
    assert generation_input["intended_category"] == "Behavioral"


def test_ten_question_backend_plan_assigns_broad_competencies(mocker):
    cfg = config("Technical", 10)
    profile = CandidateProfile(
        skills=["Node.js", "Redis", "PostgreSQL"],
        resume_summary="Built API services and asynchronous workers.",
    )
    categories = PlannerService.expected_types("Technical", 10)
    targets = PlannerService.competency_schedule(profile, cfg, categories)
    generated = [
        InterviewQuestion(
            question=f"How would you assess {target.lower()} in this backend role?",
            topic=target,
            primary_competency=target,
            difficulty="Hard",
            question_type="Technical",
            expected_concepts=[target, "Evidence", "Trade-offs"],
        )
        for target in targets
    ]
    validations = [validation(True, target, "The intended technical competency is assessed.", assessed=target) for target in targets]
    service, question_model, semantic_model = configured_service(mocker, generated, validations)

    plan = service.generate_plan(profile, cfg)

    assessed = [result.assessed_competency for result in plan.semantic_validations]
    assert len(plan.questions) == 10
    assert len(set(assessed)) >= 8
    assert len({question.topic for question in plan.questions}) >= 8
    first_input = question_model.invoke.call_args_list[0].args[0]
    assert first_input["role"] == "Backend Developer"
    assert first_input["experience"] == "Mid-level"
    assert "Redis and caching" in first_input["intended_competency"]
    assert "Redis" in semantic_model.invoke.call_args_list[0].args[0]["resume_context"]
    assert "None" in question_model.invoke.call_args_list[0].args[0]["previous_questions"]
    assert "How would you assess" in question_model.invoke.call_args_list[-1].args[0]["previous_questions"]


def test_competency_priorities_reflect_experience_level():
    profile = CandidateProfile()
    categories = ["Technical"] * 3

    junior = PlannerService.competency_schedule(profile, config("Technical", 3, "Easy"), categories)
    senior_config = config("Technical", 3, "Hard")
    senior_config.experience_level = "Senior"
    senior = PlannerService.competency_schedule(profile, senior_config, categories)

    assert junior[0] == "HTTP and REST API design"
    assert senior[0] == "System design and architectural trade-offs"


def test_semantic_duplicate_is_rejected_even_with_different_wording(mocker):
    first = question("How do you expire stale Redis keys?", category="Technical", difficulty="Hard")
    second = question("How would you keep a cache entry fresh?", category="Technical", difficulty="Hard")
    third = question("How would you choose indexes for a relational query?", category="Technical", difficulty="Hard")
    service, _, semantic_model = configured_service(
        mocker,
        [first, second, third],
        [
            validation(True, "Cache invalidation", assessed="Redis cache freshness"),
            validation(True, "Cache invalidation", duplicate=True, assessed="Redis cache freshness"),
            validation(True, "SQL indexing", assessed="SQL indexing"),
        ],
    )

    plan = service.generate_plan(CandidateProfile(), config("Technical", 2))

    assert plan.questions[0].question == first.question
    assert plan.questions[1].question == third.question
    assert semantic_model.invoke.call_count == 3
    second_review = semantic_model.invoke.call_args_list[1].args[0]
    assert "How do you expire stale Redis keys?" in second_review["existing_questions"]


def test_question_about_wrong_competency_is_repaired(mocker):
    wrong_area = question("How would you invalidate a cache?", difficulty="Hard")
    intended_area = question("How would you design an HTTP endpoint contract?", difficulty="Hard")
    service, question_model, _ = configured_service(
        mocker,
        [wrong_area, intended_area],
        [
            QuestionSemanticValidation(
                matches_intended_category=True,
                matches_intended_competency=False,
                primary_evaluation_objective="Cache invalidation",
                assessed_competency="Redis and caching",
                reason="This does not assess the assigned HTTP API competency.",
            ),
            validation(True, "HTTP and REST API design", assessed="HTTP and REST API design"),
        ],
    )

    plan = service.generate_plan(CandidateProfile(), config("Technical"))

    assert plan.questions[0].question == intended_area.question
    assert question_model.invoke.call_count == 2


def test_repetitive_plan_repair_is_bounded_and_uses_competency_fallback(mocker):
    first = question("How would you invalidate a Redis cache?", difficulty="Hard")
    duplicate = question("How would you keep a cached response fresh?", difficulty="Hard")
    service, question_model, semantic_model = configured_service(
        mocker,
        [first, duplicate, duplicate],
        [
            validation(True, "Cache invalidation", assessed="Redis and caching"),
            validation(True, "Cache invalidation", duplicate=True, assessed="Redis and caching"),
            validation(True, "Cache invalidation", duplicate=True, assessed="Redis and caching"),
        ],
        max_attempts=2,
    )

    plan = service.generate_plan(CandidateProfile(), config("Technical", 2))

    assert len(plan.questions) == 2
    assert plan.questions[1].topic != plan.questions[0].topic
    assert plan.semantic_validations[1].assessed_competency != plan.semantic_validations[0].assessed_competency
    assert question_model.invoke.call_count == 3
    assert semantic_model.invoke.call_count == 3
    assert "safe fallback" in plan.repair_warnings[0]


def test_regeneration_uses_remaining_plan_context_and_an_unused_competency(mocker):
    first = InterviewQuestion(question="How do you cache API responses?", topic="Redis and caching", primary_competency="Redis and caching", difficulty="Hard", question_type="Technical", expected_concepts=["Invalidation"])
    second = InterviewQuestion(question="How do you model relational data?", topic="SQL data modeling", primary_competency="SQL data modeling", difficulty="Hard", question_type="Technical", expected_concepts=["Indexes"])
    replacement = InterviewQuestion(question="How would you secure an API endpoint?", topic="Authentication and authorization", primary_competency="Authentication and authorization", difficulty="Hard", question_type="Technical", expected_concepts=["Identity", "Access checks"])
    service, question_model, _ = configured_service(
        mocker,
        [replacement],
        [validation(True, "Authentication and authorization", assessed="Authentication and authorization")],
    )
    plan = InterviewPlan(
        questions=[first, second],
        semantic_validations=[
            validation(True, "Redis and caching", assessed="Redis and caching"),
            validation(True, "SQL data modeling", assessed="SQL data modeling"),
        ],
    )

    service.regenerate_question(plan, CandidateProfile(), config("Technical", 2), 1)

    invocation = question_model.invoke.call_args.args[0]
    assert invocation["intended_competency"] == "HTTP and REST API design"
    assert "How do you cache API responses?" in invocation["previous_questions"]
    assert plan.questions[0] is first
    assert plan.questions[1].question == replacement.question


def test_safe_fallback_keeps_requested_question_count_when_all_candidates_invalid(mocker):
    invalid = question("How would you invalidate a cache?", difficulty="Hard")
    service, question_model, semantic_model = configured_service(
        mocker,
        [invalid] * 4,
        [validation(True, "Caching", duplicate=True, assessed="Redis and caching")] * 4,
        max_attempts=2,
    )

    plan = service.generate_plan(CandidateProfile(), config("Technical", 2))

    assert len(plan.questions) == 2
    assert question_model.invoke.call_count == 4
    assert semantic_model.invoke.call_count == 4
    assert all(item.expected_concepts for item in plan.questions)


def test_behavioral_allocation_uses_largest_remainder_and_sums_to_count():
    expected = {
        1: (1, 0, 0), 2: (1, 1, 0), 3: (1, 1, 1), 4: (2, 1, 1),
        5: (3, 1, 1), 6: (3, 2, 1), 7: (4, 2, 1), 8: (4, 2, 2),
        9: (4, 3, 2), 10: (5, 3, 2), 11: (6, 3, 2), 12: (6, 4, 2),
    }
    keys = ("HR_MOTIVATION", "EXPERIENCE_BASED", "SITUATIONAL")
    for count in range(1, 13):
        allocation = PlannerService.behavioral_category_allocation(count)
        assert tuple(allocation[key] for key in keys) == expected[count]
        assert sum(allocation.values()) == count
        assert {category: PlannerService.behavioral_category_schedule(count).count(category) for category in keys} == allocation


def test_behavioral_ten_question_plan_has_exact_subcategory_metadata(mocker):
    cfg = config("Behavioral", 10)
    profile = CandidateProfile(experience_level="Student")
    question_types = ["Behavioral"] * 10
    categories = PlannerService.behavioral_category_schedule(10)
    targets = PlannerService.competency_schedule(profile, cfg, question_types, categories)
    generated = [
        InterviewQuestion(
            question=f"Question about {target}", topic=target, primary_competency=target,
            difficulty="Hard", question_type="Behavioral", behavioral_category=subtype,
            expected_concepts=["Reasoning", "Evidence"],
        )
        for target, subtype in zip(targets, categories)
    ]
    validations = [
        validation(True, target, "The question has the assigned behavioral purpose.", assessed=target)
        for target in targets
    ]
    service, question_model, _ = configured_service(mocker, generated, validations)

    plan = service.generate_plan(profile, cfg)

    actual = [item.behavioral_category for item in plan.questions]
    assert actual.count("HR_MOTIVATION") == 5
    assert actual.count("EXPERIENCE_BASED") == 3
    assert actual.count("SITUATIONAL") == 2
    assert all(item.question_type == "Behavioral" for item in plan.questions)
    assert all(result.matches_intended_behavioral_category for result in plan.semantic_validations)
    assert question_model.invoke.call_count == 10


def test_mixed_mode_distributes_available_behavioral_subcategories_without_changing_type_ratio(mocker):
    cfg = config("Mixed", 10)
    question_types = PlannerService.expected_types("Mixed", 10)
    categories = PlannerService.behavioral_categories_for_slots(question_types)
    assert question_types.count("Technical") == 6
    assert question_types.count("Behavioral") == 4
    assert categories.count("HR_MOTIVATION") == 2
    assert categories.count("EXPERIENCE_BASED") == 1
    assert categories.count("SITUATIONAL") == 1

    targets = PlannerService.competency_schedule(CandidateProfile(), cfg, question_types, categories)
    generated = []
    validations = []
    for index, (kind, subtype, target) in enumerate(zip(question_types, categories, targets)):
        generated.append(InterviewQuestion(
            question=f"Plan item {index}: {target}", topic=target, primary_competency=target,
            difficulty="Hard", question_type=kind, behavioral_category=subtype,
            expected_concepts=["Reasoning"],
        ))
        validations.append(validation(True, target, "Intent matches the assigned slot.", assessed=target))
    service, _, _ = configured_service(mocker, generated, validations)

    plan = service.generate_plan(CandidateProfile(), cfg)

    assert [item.question_type for item in plan.questions] == question_types
    assert [item.behavioral_category for item in plan.questions] == categories


def test_repetitive_behavioral_question_across_subtypes_is_repaired(mocker):
    generated = [
        question("What motivates you in your work?", category="Behavioral", behavioral_category="HR_MOTIVATION"),
        question("Tell me about a time you felt motivated to help a team succeed.", category="Behavioral", behavioral_category="EXPERIENCE_BASED"),
        question("Tell me about a mistake you made and what you learned.", category="Behavioral", behavioral_category="EXPERIENCE_BASED"),
        question("Two tasks are urgent. How would you prioritize them?", category="Behavioral", behavioral_category="SITUATIONAL"),
    ]
    # Three slots allocate HR, experience, and situational. The experience
    # question overlaps semantically with HR motivation and must be repaired.
    service, _, semantic_model = configured_service(
        mocker,
        generated,
        [
            validation(True, "Motivation", assessed="Motivation"),
            validation(True, "Motivation", duplicate=True, assessed="Motivation"),
            validation(True, "Learning from mistakes", assessed="Learning from mistakes"),
            validation(True, "Prioritization", assessed="Prioritization"),
        ],
    )
    cfg = config("Behavioral", 3)

    plan = service.generate_plan(CandidateProfile(), cfg)

    assert [item.behavioral_category for item in plan.questions] == ["HR_MOTIVATION", "EXPERIENCE_BASED", "SITUATIONAL"]
    assert plan.questions[1].question == generated[2].question
    assert semantic_model.invoke.call_count == 4
    second_review = semantic_model.invoke.call_args_list[1].args[0]
    assert "What motivates you in your work?" in second_review["existing_questions"]


def test_regeneration_preserves_experience_subcategory(mocker):
    cfg = config("Behavioral", 3)
    categories = PlannerService.behavioral_category_schedule(3)
    types = ["Behavioral"] * 3
    targets = PlannerService.competency_schedule(CandidateProfile(), cfg, types, categories)
    questions = [
        InterviewQuestion(
            question=f"Original {subtype} question {index}", topic=target, primary_competency=target,
            difficulty="Hard", question_type="Behavioral", behavioral_category=subtype,
            expected_concepts=["Evidence"],
        )
        for index, (subtype, target) in enumerate(zip(categories, targets))
    ]
    validations = [validation(True, target, "Valid category.", assessed=target) for target in targets]
    replacement = InterviewQuestion(
        question="Tell me about a time you took initiative on a team project.",
        topic=targets[1], primary_competency=targets[1], difficulty="Hard",
        question_type="Behavioral", behavioral_category="EXPERIENCE_BASED",
        expected_concepts=["Initiative", "Outcome"],
    )
    service, question_model, _ = configured_service(
        mocker, [replacement], [validation(True, targets[1], "Past action and outcome.", assessed=targets[1])]
    )
    plan = InterviewPlan(questions=questions, semantic_validations=validations)

    service.regenerate_question(plan, CandidateProfile(), cfg, 1)

    assert plan.questions[1].behavioral_category == "EXPERIENCE_BASED"
    assert plan.questions[0] is questions[0] and plan.questions[2] is questions[2]
    assert question_model.invoke.call_args.args[0]["behavioral_category"] == "EXPERIENCE_BASED"


def test_early_career_experience_prompt_allows_nonprofessional_examples_and_never_invents(mocker):
    cfg = config("Behavioral", 3)
    cfg.experience_level = "Student"
    categories = PlannerService.behavioral_category_schedule(3)
    types = ["Behavioral"] * 3
    targets = PlannerService.competency_schedule(CandidateProfile(), cfg, types, categories)
    generated = [
        InterviewQuestion(question=f"Question {i}", topic=target, primary_competency=target,
            difficulty="Hard", question_type="Behavioral", behavioral_category=subtype,
            expected_concepts=["Evidence"])
        for i, (subtype, target) in enumerate(zip(categories, targets))
    ]
    validations = [validation(True, target, "Valid behavioral subtype.", assessed=target) for target in targets]
    service, question_model, _ = configured_service(mocker, generated, validations)

    service.generate_plan(CandidateProfile(), cfg)

    experience_input = question_model.invoke.call_args_list[1].args[0]
    assert experience_input["behavioral_category"] == "EXPERIENCE_BASED"
    guidance = experience_input["category_guidance"].casefold()
    assert "college project" in guidance
    assert "never invent" in guidance
    assert experience_input["candidate_experience"] == "Not provided"


def test_plan_rejects_missing_semantic_validation():
    plan = InterviewPlan(questions=[question(difficulty="Hard")])
    with pytest.raises(ValueError, match="semantic validation"):
        service = PlannerService.__new__(PlannerService)
        service.validate_plan(plan, config())

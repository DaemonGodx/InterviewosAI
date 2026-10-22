from dataclasses import dataclass
from math import floor
from typing import Any, Callable, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from app.config import GROQ_API_KEY
from app.models.schemas import (
    CandidateProfile,
    InterviewConfig,
    InterviewPlan,
    InterviewQuestion,
    GeneratedQuestionReview,
    QuestionSemanticValidation,
)


class PlanValidationError(ValueError):
    """Raised when an interview plan cannot be made internally consistent."""


class QuestionSemanticError(ValueError):
    """Raised when a proposed question fails semantic category validation."""


@dataclass
class QuestionBuildResult:
    question: InterviewQuestion
    semantic_validation: QuestionSemanticValidation
    repair_warning: Optional[str] = None


class PlannerService:
    """Build and semantically validate a complete plan before interview start."""

    MAX_ATTEMPTS = 3

    def __init__(self, llm: Any = None, max_attempts: int = MAX_ATTEMPTS):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.llm = llm or ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=GROQ_API_KEY,
            temperature=0.2,
        )
        self.max_attempts = max_attempts

    @staticmethod
    def expected_types(interview_type: str, count: int) -> list[str]:
        """Return a deterministic category schedule, with 60/40 Mixed coverage."""
        if count < 1:
            raise PlanValidationError("An interview must contain at least one question.")
        if interview_type == "Technical":
            return ["Technical"] * count
        if interview_type == "Behavioral":
            return ["Behavioral"] * count
        if interview_type != "Mixed":
            raise PlanValidationError(f"Unsupported interview type: {interview_type}")
        if count == 1:
            return ["Technical"]

        technical_count = max(1, min(count - 1, floor(count * 0.6 + 0.5)))
        schedule = []
        previous_technical_slots = 0
        for completed in range(1, count + 1):
            current_technical_slots = floor(completed * technical_count / count + 0.5)
            if current_technical_slots > previous_technical_slots:
                schedule.append("Technical")
            else:
                schedule.append("Behavioral")
            previous_technical_slots = current_technical_slots
        return schedule

    def generate_plan(
        self, profile: CandidateProfile, config: InterviewConfig,
        progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    ) -> InterviewPlan:
        expected = self.expected_types(config.interview_type, config.question_count)
        behavioral_categories = self.behavioral_categories_for_slots(expected)
        competency_targets = self.competency_schedule(profile, config, expected, behavioral_categories)
        questions = []
        semantic_validations = []
        warnings = []

        for index, (category, competency, behavioral_category) in enumerate(
            zip(expected, competency_targets, behavioral_categories), start=1
        ):
            if progress_callback:
                progress_callback(index, config.question_count, category, competency)
            built = self._build_question(
                profile, config, category, index, competency,
                existing_questions=questions,
                behavioral_category=behavioral_category,
            )
            questions.append(built.question)
            semantic_validations.append(built.semantic_validation)
            if built.repair_warning:
                warnings.append(f"Question {index}: {built.repair_warning}")

        plan = InterviewPlan(
            questions=questions,
            semantic_validations=semantic_validations,
            repair_warnings=warnings,
        )
        self.validate_plan(plan, config)
        return plan

    def regenerate_question(self, plan: InterviewPlan, profile: CandidateProfile, config: InterviewConfig, index: int) -> QuestionBuildResult:
        """Regenerate one slot while preserving all other plan entries."""
        if index < 0 or index >= len(plan.questions):
            raise IndexError("Question number is outside the plan.")
        expected = self.expected_types(config.interview_type, config.question_count)
        if len(plan.questions) != config.question_count:
            raise PlanValidationError("The current plan has the wrong number of questions.")
        if len(plan.semantic_validations) != len(plan.questions):
            raise PlanValidationError("Each planned question must have a semantic validation result.")

        existing_questions = [question for i, question in enumerate(plan.questions) if i != index]
        existing_competencies = {
            self._normalize_competency(plan.semantic_validations[i].assessed_competency)
            for i in range(len(plan.semantic_validations)) if i != index
        }
        slot_behavioral_categories = self.behavioral_categories_for_slots(expected)
        original_behavioral_category = (
            plan.questions[index].behavioral_category or slot_behavioral_categories[index]
        )
        slot_behavioral_categories[index] = original_behavioral_category
        targets = self.competency_schedule(profile, config, expected, slot_behavioral_categories)
        available = [
            targets[i] for i, category in enumerate(expected)
            if category == expected[index]
            and slot_behavioral_categories[i] == original_behavioral_category
            and self._normalize_competency(targets[i]) not in existing_competencies
        ]
        target = available[0] if available else targets[index]
        built = self._build_question(
            profile, config, expected[index], index + 1, target,
            existing_questions=existing_questions,
            behavioral_category=original_behavioral_category,
        )
        candidate_plan = plan.model_copy(deep=True)
        candidate_plan.questions[index] = built.question
        candidate_plan.semantic_validations[index] = built.semantic_validation
        self._replace_warning(candidate_plan, index, built.repair_warning)
        self.validate_plan(candidate_plan, config)

        plan.questions[index] = built.question
        plan.semantic_validations[index] = built.semantic_validation
        self._replace_warning(plan, index, built.repair_warning)
        return built

    def edit_question(self, plan: InterviewPlan, config: InterviewConfig, index: int, replacement_text: str) -> QuestionSemanticValidation:
        """Validate and apply a text edit; invalid edits leave the plan untouched."""
        if index < 0 or index >= len(plan.questions):
            raise IndexError("Question number is outside the plan.")
        replacement_text = replacement_text.strip()
        if not replacement_text:
            raise QuestionSemanticError("The replacement question cannot be empty.")

        expected = self.expected_types(config.interview_type, config.question_count)[index]
        current = plan.questions[index]
        candidate = current.model_copy(update={"question": replacement_text})
        existing_questions = [question for i, question in enumerate(plan.questions) if i != index]
        validation = self.validate_question_semantics(
            candidate, expected, config.target_role,
            intended_competency=current.primary_competency or current.topic,
            existing_questions=existing_questions,
            config=config,
            intended_behavioral_category=current.behavioral_category,
        )
        if not validation.matches_intended_category:
            raise QuestionSemanticError(validation.reason)
        if not validation.matches_intended_competency:
            raise QuestionSemanticError("The edited question no longer assesses its assigned competency.")
        if current.behavioral_category and not validation.matches_intended_behavioral_category:
            raise QuestionSemanticError("The edit no longer matches its assigned behavioral category.")
        if validation.is_semantic_duplicate:
            raise QuestionSemanticError(validation.duplicate_reason or "The edit duplicates another question's primary competency.")
        if current.question_type != expected:
            raise QuestionSemanticError("The existing question metadata does not match its intended category.")

        candidate_plan = plan.model_copy(deep=True)
        candidate_plan.questions[index] = candidate
        candidate_plan.semantic_validations[index] = validation
        self._replace_warning(candidate_plan, index, None)
        self.validate_plan(candidate_plan, config)

        plan.questions[index] = candidate
        plan.semantic_validations[index] = validation
        self._replace_warning(plan, index, None)
        return validation

    def validate_plan(self, plan: InterviewPlan, config: InterviewConfig) -> None:
        """Check the full plan's count, schedule, metadata, rubric, and validations."""
        expected = self.expected_types(config.interview_type, config.question_count)
        expected_behavioral_categories = self.behavioral_categories_for_slots(expected)
        if len(plan.questions) != config.question_count:
            raise PlanValidationError(f"Plan has {len(plan.questions)} questions; expected {config.question_count}.")
        if len(plan.semantic_validations) != len(plan.questions):
            raise PlanValidationError("Each planned question must have a semantic validation result.")

        for index, (question, category, validation) in enumerate(
            zip(plan.questions, expected, plan.semantic_validations), start=1
        ):
            if question.question_type != category:
                raise PlanValidationError(f"Question {index} is labelled {question.question_type}; expected {category}.")
            intended_behavioral_category = expected_behavioral_categories[index - 1]
            if question.behavioral_category != intended_behavioral_category:
                raise PlanValidationError(
                    f"Question {index} has behavioral category {question.behavioral_category}; expected {intended_behavioral_category}."
                )
            if question.difficulty != config.starting_difficulty:
                raise PlanValidationError(f"Question {index} has inconsistent difficulty metadata.")
            if not question.question.strip():
                raise PlanValidationError(f"Question {index} is empty.")
            if not question.expected_concepts or any(not concept.strip() for concept in question.expected_concepts):
                raise PlanValidationError(f"Question {index} must include non-empty expected concepts.")
            if not validation.matches_intended_category:
                raise PlanValidationError(f"Question {index} has not passed semantic category validation.")
            if not validation.matches_intended_competency:
                raise PlanValidationError(f"Question {index} has not passed semantic competency validation.")
            if intended_behavioral_category and not validation.matches_intended_behavioral_category:
                raise PlanValidationError(f"Question {index} has not passed behavioral-category validation.")
            if validation.is_semantic_duplicate:
                raise PlanValidationError(f"Question {index} duplicates another question's primary competency.")
            if not validation.assessed_competency.strip():
                raise PlanValidationError(f"Question {index} is missing its normalized assessed competency.")
            if not validation.primary_evaluation_objective.strip() or not validation.reason.strip():
                raise PlanValidationError(f"Question {index} is missing its semantic validation rationale.")

    def validate_question_semantics(
        self, question: InterviewQuestion, intended_category: str, role: str,
        intended_competency: Optional[str] = None,
        existing_questions: Optional[list[InterviewQuestion]] = None,
        config: Optional[InterviewConfig] = None,
        profile: Optional[CandidateProfile] = None,
        intended_behavioral_category: Optional[str] = None,
    ) -> QuestionSemanticValidation:
        existing_questions = existing_questions or []
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strict interviewer-question semantic reviewer. Assess the question's PRIMARY evaluation objective, not just its wording or supplied label.

Technical means the candidate is primarily evaluated on technical knowledge, implementation, debugging, architecture, problem-solving, or engineering trade-offs.
Behavioral means the candidate is primarily evaluated on communication, teamwork, conflict resolution, leadership, ownership, adaptability, motivation, professional judgment, or learning.

Behavioral subtypes are distinct: HR_MOTIVATION assesses identity, motivation, role/company interest, self-awareness, or career reasoning; EXPERIENCE_BASED assesses an actual past action and outcome; SITUATIONAL assesses a hypothetical decision and reasoning. A technical system-design question does not become Behavioral because it is framed as a past story or carries a Behavioral label.

A question that starts with 'Tell me about a time' can still be Technical if the expected answer is mainly a system design or implementation explanation. A question may mention a technical project and still be Behavioral if the expected evidence is the candidate's actions, communication, judgment, and outcome. Accept situational questions when their answer is judged on behavior rather than the technical solution.

Return whether the primary evaluation objective matches the intended category and the assigned competency. For Behavioral questions also determine whether it matches the intended behavioral subtype. Mark matches_intended_competency false if the actual question tests a different area, even if that area is generally relevant to the role. Mark it true for a reasonable equivalent framing of the assigned competency. Name the actual primary objective and explain your decision. Also give a short normalized assessed_competency (for example, 'SQL indexing and query planning') that captures the underlying knowledge or behavior, independent of wording. Compare against the other questions by the competency they assess, not lexical similarity. Set is_semantic_duplicate true when substantially the same competency is tested, even with different phrasing; otherwise false. Explain any duplicate in duplicate_reason. Do not treat a related but distinct engineering area as a duplicate. Be conservative when the question mixes categories: use the dominant assessment objective."""),
            ("human", "Role: {role}\nExperience: {experience}\nInterview type: {interview_type}\nResume context: {resume_context}\nIntended category: {intended_category}\nIntended behavioral subtype: {intended_behavioral_category}\nIntended competency: {intended_competency}\nSupplied category label: {supplied_category}\nSupplied behavioral subtype: {supplied_behavioral_category}\nQuestion: {question}\nOther planned questions (1-based): {existing_questions}"),
        ])
        chain = prompt | self.llm.with_structured_output(QuestionSemanticValidation)
        result = chain.invoke({
            "role": role,
            "experience": config.experience_level if config else "Not provided",
            "interview_type": config.interview_type if config else intended_category,
            "resume_context": self._resume_context(profile) if profile else "Not provided",
            "intended_category": intended_category,
            "intended_behavioral_category": intended_behavioral_category or "Not applicable",
            "intended_competency": intended_competency or "Use the question's primary competency",
            "supplied_category": question.question_type,
            "supplied_behavioral_category": question.behavioral_category or "Not applicable",
            "question": question.question,
            "existing_questions": "\n".join(
                f"{i}. [{item.question_type}; {item.behavioral_category or 'n/a'}; {item.primary_competency or item.topic}] {item.question}"
                for i, item in enumerate(existing_questions, start=1)
            ) or "None",
        })
        return self._coerce_result(QuestionSemanticValidation, result)

    def _build_question(
        self, profile: CandidateProfile, config: InterviewConfig, intended_category: str,
        question_number: int, intended_competency: str,
        existing_questions: Optional[list[InterviewQuestion]] = None,
        behavioral_category: Optional[str] = None,
    ) -> QuestionBuildResult:
        existing_questions = existing_questions or []
        failures = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                generated = self._generate_candidate(
                    profile, config, intended_category, question_number, intended_competency,
                    existing_questions,
                    behavioral_category,
                )
                question = generated.question
                validation = generated.semantic_validation

                if question.question_type != intended_category:
                    failures.append(f"attempt {attempt}: metadata said {question.question_type}, expected {intended_category}")
                    continue
                if question.behavioral_category != behavioral_category:
                    failures.append(f"attempt {attempt}: behavioral category metadata did not match the assigned slot")
                    continue
                if not validation.matches_intended_category:
                    failures.append(f"attempt {attempt}: {validation.reason}")
                    continue
                if not validation.matches_intended_competency:
                    failures.append(f"attempt {attempt}: question did not assess the assigned competency")
                    continue
                if behavioral_category and (
                    question.behavioral_category != behavioral_category
                    or not validation.matches_intended_behavioral_category
                ):
                    failures.append(f"attempt {attempt}: question did not match the assigned behavioral subtype")
                    continue
                if validation.is_semantic_duplicate:
                    failures.append(f"attempt {attempt}: {validation.duplicate_reason or 'semantic duplicate of an existing question'}")
                    continue
                if not validation.assessed_competency.strip():
                    failures.append(f"attempt {attempt}: semantic reviewer omitted the assessed competency")
                    continue
                if not question.expected_concepts or any(not concept.strip() for concept in question.expected_concepts):
                    failures.append(f"attempt {attempt}: expected concepts were missing")
                    continue

                question.difficulty = config.starting_difficulty
                question.primary_competency = question.primary_competency or question.topic
                return QuestionBuildResult(question=question, semantic_validation=validation)
            except Exception as error:
                # Provider exceptions may include request configuration; keep
                # repair reports safe to display and persist.
                failures.append(f"attempt {attempt}: model generation or semantic validation failed")

        fallback = self._safe_fallback(
            intended_category, config, question_number, intended_competency, behavioral_category
        )
        fallback_validation = QuestionSemanticValidation(
            matches_intended_category=True,
            matches_intended_competency=True,
            matches_intended_behavioral_category=bool(behavioral_category),
            primary_evaluation_objective=f"{intended_category} assessment of {intended_competency}",
            reason="Deterministic category-appropriate fallback used after bounded model validation attempts.",
            assessed_competency=intended_competency,
            is_semantic_duplicate=False,
        )
        details = "; ".join(failures[-self.max_attempts:])
        return QuestionBuildResult(
            question=fallback,
            semantic_validation=fallback_validation,
            repair_warning=f"Model question did not pass validation after {self.max_attempts} attempt(s); using a safe fallback. {details}",
        )

    def _generate_candidate(
        self, profile: CandidateProfile, config: InterviewConfig, intended_category: str,
        question_number: int, intended_competency: str,
        existing_questions: list[InterviewQuestion],
        behavioral_category: Optional[str] = None,
    ) -> GeneratedQuestionReview:
        if intended_category == "Behavioral":
            category_guidance = self._behavioral_guidance(behavioral_category, profile, config)
        else:
            category_guidance = """The main task must assess technical knowledge, implementation, debugging, architecture, problem-solving, or engineering trade-offs. It may ask about a past technical experience, but the answer must primarily demonstrate technical reasoning rather than interpersonal behavior."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert interviewer and semantic reviewer. Generate exactly one concise question for slot {question_number} of {question_count}, then include its semantic review in the same structured response.
Its intended category is {intended_category}. The question's primary evaluation objective must match that category. Its behavioral subtype is {behavioral_category}.
{category_guidance}
Set question_type metadata to the intended category exactly, behavioral_category metadata to the assigned subtype (or null for Technical), and difficulty to {difficulty}. The intended primary competency for this slot is {intended_competency}; keep the question focused on it and use matching topic/primary_competency metadata. Include a short list of concrete expected concepts. Avoid assessing the same underlying competency as these questions, even if you phrase it differently: {previous_questions}.

In semantic_validation, review the actual expected answer objective, not wording or labels. Technical means technical knowledge, implementation, debugging, architecture, problem-solving, or engineering trade-offs. Behavioral means communication, teamwork, conflict resolution, leadership, ownership, adaptability, motivation, professional judgment, or learning. Past-experience wording may still be Technical if the answer is judged on design or implementation. For Behavioral, verify the assigned HR_MOTIVATION, EXPERIENCE_BASED, or SITUATIONAL subtype. Return category, competency, and subtype match flags; name the primary evaluation objective; provide a normalized assessed_competency; and compare it against all other questions for semantic duplicates. Give a concise reason and duplicate_reason when relevant. Return both question and semantic_validation."""),
            ("human", "Role: {role}\nExperience level: {experience}\nResume experience: {candidate_experience}\nResume summary: {summary}\nSkills: {skills}\nProjects: {projects}\nRelevant technologies: {technologies}"),
        ])
        chain = prompt | self.llm.with_structured_output(GeneratedQuestionReview)
        result = chain.invoke({
            "question_number": question_number,
            "question_count": config.question_count,
            "intended_category": intended_category,
            "category_guidance": category_guidance,
            "behavioral_category": behavioral_category or "Not applicable",
            "intended_competency": intended_competency,
            "difficulty": config.starting_difficulty,
            "previous_questions": "\n".join(
                f"{i}. [{item.primary_competency or item.topic}] {item.question}"
                for i, item in enumerate(existing_questions, start=1)
            ) or "None",
            # Keep explicit reviewer context available in the same structured
            # call; this also makes the review inputs auditable in tests/logs.
            "existing_questions": "\n".join(
                f"{i}. [{item.question_type}; {item.behavioral_category or 'n/a'}; {item.primary_competency or item.topic}] {item.question}"
                for i, item in enumerate(existing_questions, start=1)
            ) or "None",
            "resume_context": self._resume_context(profile),
            "interview_type": config.interview_type,
            "supplied_category": intended_category,
            "supplied_behavioral_category": behavioral_category or "Not applicable",
            "role": config.target_role,
            "experience": config.experience_level,
            "candidate_experience": "; ".join(profile.experience) or "Not provided",
            "summary": profile.resume_summary or "No summary provided.",
            "skills": ", ".join(profile.skills) or "Not provided",
            "projects": "; ".join(project.name + ": " + project.description for project in profile.projects) or "Not provided",
            "technologies": ", ".join(profile.interviewable_technologies) or "General role-related topics",
        })
        return self._coerce_result(GeneratedQuestionReview, result)

    @staticmethod
    def _safe_fallback(category: str, config: InterviewConfig, question_number: int, competency: str = "Technical problem-solving", behavioral_category: Optional[str] = None) -> InterviewQuestion:
        if category == "Technical":
            fallback_questions = {
                "HTTP and REST API design": ("How would you design a REST endpoint for a resource with filtering and pagination? Explain its contract and failure behavior.", ["HTTP semantics", "Pagination", "Validation and errors"]),
                "JavaScript and Node.js": ("In a Node.js service, how would you structure a request handler so input validation and response shaping remain reliable?", ["Runtime behavior", "Input validation", "Module boundaries"]),
                "Asynchronous programming and error handling": ("How would you handle a request that depends on several asynchronous operations when one can fail or time out?", ["Async control flow", "Timeouts", "Error propagation"]),
                "MongoDB, SQL, indexing, and data modeling": ("How would you choose a data model and indexes for a query-heavy feature, and verify that the queries scale?", ["Data modeling", "Index selection", "Query plans"]),
                "Authentication and authorization": ("How would you enforce authentication and resource-level authorization for an API endpoint?", ["Identity validation", "Authorization checks", "Failure handling"]),
                "Redis and caching": ("How would you decide whether to cache a frequently read value, and keep the cached value acceptably fresh?", ["Cache key design", "Invalidation", "Staleness trade-offs"]),
                "Background jobs and event-driven architecture": ("How would you process a slow task asynchronously while making retries safe?", ["Queue design", "Idempotency", "Retry policy"]),
                "Concurrency and distributed systems": ("How would you prevent two concurrent requests from corrupting the same shared resource?", ["Race conditions", "Consistency", "Coordination"]),
                "Security, reliability, and observability": ("How would you make a service endpoint safer and easier to diagnose in production?", ["Threat boundaries", "Reliability", "Metrics and logs"]),
                "Debugging and production troubleshooting": ("A production endpoint becomes intermittently slow. How would you isolate the cause and confirm a fix?", ["Evidence gathering", "Hypothesis testing", "Regression prevention"]),
                "System design and architectural trade-offs": ("Design a service for a stated workload and explain the main architectural trade-offs and failure modes.", ["Requirements", "Architecture", "Trade-offs"]),
            }
            text, concepts = fallback_questions.get(
                competency,
                (f"How would you approach {competency.lower()} in a {config.target_role} system? Explain your reasoning and trade-offs.", [competency, "Reasoning", "Trade-offs"]),
            )
            return InterviewQuestion(
                question=text,
                topic=competency,
                primary_competency=competency,
                difficulty=config.starting_difficulty,
                question_type="Technical",
                behavioral_category=None,
                expected_concepts=concepts,
            )

        if behavioral_category == "HR_MOTIVATION":
            behavioral_questions = {
                "Professional introduction and career narrative": ("Tell me about yourself and the experiences that shaped your interest in this role.", ["Relevant background", "Motivation", "Career direction"]),
                "Role motivation and alignment": (f"What interests you about this {config.target_role} role?", ["Role understanding", "Motivation", "Alignment"]),
                "Company motivation": ("What interests you about joining this organization?", ["Motivation", "Role/company fit", "Reasoning"]),
                "Self-awareness and strengths": ("What strength do you rely on most at work or in your studies, and what are you working to improve?", ["Self-awareness", "Example", "Growth area"]),
                "Motivation and values": ("What kind of work motivates you, and why?", ["Motivators", "Values", "Reflection"]),
                "Career goals": ("What would you like to learn or accomplish in the next few years?", ["Goals", "Learning", "Role alignment"]),
            }
        elif behavioral_category == "SITUATIONAL":
            behavioral_questions = {
                "Prioritization and judgment": ("Two urgent tasks have the same deadline. How would you decide what to do first and communicate the trade-off?", ["Prioritization", "Communication", "Trade-off reasoning"]),
                "Conflict resolution": ("If a teammate repeatedly missed commitments, how would you address it?", ["Direct communication", "Understanding context", "Follow-through"]),
                "Manager disagreement": ("If you disagreed with your manager about an approach, how would you raise your concern and work toward a decision?", ["Respectful challenge", "Evidence", "Alignment"]),
                "Teamwork and collaboration": ("If a teammate was struggling to complete a shared task, what would you do?", ["Support", "Communication", "Shared outcome"]),
                "Ownership and follow-through": ("If you noticed a problem outside your assigned task, how would you decide whether and how to act?", ["Judgment", "Initiative", "Coordination"]),
            }
        else:
            behavioral_questions = {
            "Communication and collaboration": ("Tell me about a time you worked with others to deliver a difficult piece of work. What did you contribute?", ["Clear communication", "Collaboration", "Outcome"]),
                "Ownership and follow-through": ("Describe a time you took ownership of a problem and followed it through to an outcome.", ["Initiative", "Follow-through", "Impact"]),
                "Conflict resolution": ("Describe a disagreement with a teammate and how you resolved it.", ["Listening", "Constructive communication", "Resolution"]),
                "Learning and adaptability": ("Tell me about a mistake or difficult feedback that changed how you work.", ["Accountability", "Reflection", "Learning"]),
                "Prioritization and judgment": ("Tell me about a time you handled competing priorities. How did you decide what to do?", ["Judgment", "Communication", "Trade-offs"]),
                "Leadership and influence": ("Describe a time you influenced a team outcome without having formal authority.", ["Initiative", "Influence", "Team outcome"]),
            }
        text, concepts = behavioral_questions.get(
            competency,
            ("Tell me about a difficult challenge you handled. What did you do, and what was the outcome?", ["Ownership", "Communication", "Outcome"]),
        )
        topic = competency
        return InterviewQuestion(
            question=text,
            topic=topic,
            primary_competency=topic,
            difficulty=config.starting_difficulty,
            question_type="Behavioral",
            behavioral_category=behavioral_category or "EXPERIENCE_BASED",
            expected_concepts=concepts,
        )

    @staticmethod
    def behavioral_category_allocation(count: int) -> dict[str, int]:
        """Largest-remainder allocation of the behavioral 50/30/20 mix."""
        if count < 0:
            raise ValueError("Behavioral question count cannot be negative.")
        weights = [("HR_MOTIVATION", 0.50), ("EXPERIENCE_BASED", 0.30), ("SITUATIONAL", 0.20)]
        exact = {name: count * weight for name, weight in weights}
        allocation = {name: int(exact[name]) for name, _ in weights}
        remaining = count - sum(allocation.values())
        tie_order = {name: index for index, (name, _) in enumerate(weights)}
        remainder_order = sorted(
            (name for name, _ in weights),
            key=lambda name: (-(exact[name] - allocation[name]), tie_order[name]),
        )
        for name in remainder_order[:remaining]:
            allocation[name] += 1
        return allocation

    @classmethod
    def behavioral_category_schedule(cls, count: int) -> list[str]:
        """Spread the exact largest-remainder allocation across the plan."""
        allocation = cls.behavioral_category_allocation(count)
        scheduled = {name: 0 for name in allocation}
        result = []
        for slot in range(1, count + 1):
            choice = max(
                allocation,
                key=lambda name: (
                    allocation[name] * slot / count - scheduled[name] if count else 0,
                    -list(allocation).index(name),
                ),
            )
            result.append(choice)
            scheduled[choice] += 1
        return result

    @classmethod
    def behavioral_categories_for_slots(cls, question_types: list[str]) -> list[Optional[str]]:
        schedule = iter(cls.behavioral_category_schedule(question_types.count("Behavioral")))
        return [next(schedule) if question_type == "Behavioral" else None for question_type in question_types]

    @staticmethod
    def _behavioral_guidance(category: Optional[str], profile: CandidateProfile, config: InterviewConfig) -> str:
        experience = (config.experience_level + " " + " ".join(profile.experience)).casefold()
        early_career = any(term in experience for term in ("student", "intern", "fresher", "junior", "entry", "0-2", "no professional"))
        if early_career:
            settings = "The candidate may use a job, college project, coursework, team activity, or extracurricular setting. Do not assume they have professional experience, and never invent personal history, achievements, employers, or projects."
        else:
            settings = "The candidate may draw on professional, project, coursework, or extracurricular examples when relevant; never presume or invent personal history, achievements, employers, or projects."
        if category == "HR_MOTIVATION":
            return "Ask a natural, conversational HR or hiring-manager question about the candidate's background, motivation, role/company interest, self-awareness, or career goals. Avoid technical knowledge checks and avoid inventing facts about the company. " + settings
        if category == "EXPERIENCE_BASED":
            return "Ask about a real past experience, such as a mistake/failure and learning, teamwork, disagreement, critical feedback, initiative, setback, or competing priorities. Ask what the candidate personally did and what happened. " + settings
        if category == "SITUATIONAL":
            return "Ask a realistic hypothetical workplace question. Evaluate the candidate's decisions, priorities, communication, and reasoning in the scenario, not technical design. Keep these situational prompts a minority according to the assigned plan allocation."
        return "Ask a natural behavioral question and evaluate professional conduct rather than technical implementation. " + settings

    @classmethod
    def competency_schedule(
        cls, profile: CandidateProfile, config: InterviewConfig, categories: list[str],
        behavioral_categories: Optional[list[Optional[str]]] = None,
    ) -> list[str]:
        """Assign a varied, role-aware competency to each scheduled category."""
        role = config.target_role.casefold()
        if any(word in role for word in ("backend", "back-end", "server", "api")):
            technical = [
                "HTTP and REST API design", "JavaScript and Node.js",
                "Asynchronous programming and error handling",
                "MongoDB, SQL, indexing, and data modeling", "Authentication and authorization",
                "Redis and caching", "Background jobs and event-driven architecture",
                "Concurrency and distributed systems", "Security, reliability, and observability",
                "Debugging and production troubleshooting", "System design and architectural trade-offs",
            ]
        elif any(word in role for word in ("frontend", "front-end", "web ui")):
            technical = ["JavaScript and browser fundamentals", "UI state and component design", "HTTP and API integration", "Asynchronous programming and error handling", "Accessibility and semantic HTML", "Performance and rendering", "Testing and debugging", "Security and reliability", "Frontend architecture and trade-offs"]
        elif any(word in role for word in ("data", "analytics", "database")):
            technical = ["SQL and query planning", "Data modeling and quality", "Python data processing", "Statistics and experiment reasoning", "Pipeline reliability", "Distributed data processing", "Security and governance", "Debugging and observability", "System design and trade-offs"]
        elif any(word in role for word in ("machine learning", "ml", " ai ")):
            technical = ["Model evaluation and metrics", "Data quality and leakage", "Feature engineering", "Training and inference systems", "Deployment and monitoring", "Experiment design", "Scalability and latency", "Security and reliability", "ML system trade-offs"]
        else:
            technical = ["Role-specific fundamentals", "Implementation and debugging", "Data modeling and persistence", "Reliability and error handling", "Security and access control", "Testing and quality", "System design and trade-offs"]

        resume_terms = " ".join([
            *profile.skills, *profile.frameworks, *profile.databases,
            *profile.interviewable_technologies, *profile.experience,
            *(f"{project.name} {project.description} {' '.join(project.technologies)}" for project in profile.projects),
        ]).casefold()
        # Put candidate technologies first when they are relevant, while retaining
        # breadth across competencies rather than repeatedly testing one stack item.
        stack_competencies = []
        if any(term in resume_terms for term in ("redis", "cache")):
            stack_competencies.append("Redis and caching")
        if any(term in resume_terms for term in ("mongodb", "mongo", "sql", "postgres", "mysql")):
            stack_competencies.append("MongoDB, SQL, indexing, and data modeling")
        if any(term in resume_terms for term in ("node", "javascript", "express")):
            stack_competencies.append("JavaScript and Node.js")
        if any(term in resume_terms for term in ("kafka", "rabbitmq", "queue", "event")):
            stack_competencies.append("Background jobs and event-driven architecture")
        technical = list(dict.fromkeys(stack_competencies + technical))
        level = config.experience_level.casefold()
        if any(term in level for term in ("senior", "staff", "principal", "lead", "5+", "7+")):
            advanced = ["System design and architectural trade-offs", "Concurrency and distributed systems", "Security, reliability, and observability"]
            technical = list(dict.fromkeys(advanced + technical))
        elif any(term in level for term in ("student", "intern", "fresher", "junior", "entry", "0-2")):
            fundamentals = ["HTTP and REST API design", "JavaScript and Node.js", "Asynchronous programming and error handling", "Debugging and production troubleshooting"]
            technical = list(dict.fromkeys(fundamentals + technical))

        behavioral_categories = behavioral_categories or cls.behavioral_categories_for_slots(categories)
        behavioral_competencies = {
            "HR_MOTIVATION": [
                "Professional introduction and career narrative", "Role motivation and alignment",
                "Company motivation", "Self-awareness and strengths", "Career goals",
            ],
            "EXPERIENCE_BASED": [
                "Communication and collaboration", "Ownership and follow-through", "Conflict resolution",
                "Learning and adaptability", "Initiative and setbacks", "Prioritization and judgment",
            ],
            "SITUATIONAL": [
                "Prioritization and judgment", "Conflict resolution", "Manager disagreement",
                "Teamwork and collaboration", "Ownership and follow-through",
            ],
        }
        schedule = []
        category_positions = {"Technical": 0, "Behavioral": 0}
        subtype_positions = {name: 0 for name in behavioral_competencies}
        for index, category in enumerate(categories):
            if category == "Technical":
                choices = technical
                pos = category_positions[category]
                schedule.append(choices[pos % len(choices)])
                category_positions[category] += 1
            else:
                subtype = behavioral_categories[index] or "EXPERIENCE_BASED"
                choices = behavioral_competencies[subtype]
                pos = subtype_positions[subtype]
                schedule.append(choices[pos % len(choices)])
                subtype_positions[subtype] += 1
        return schedule

    @staticmethod
    def _normalize_competency(value: str) -> str:
        return " ".join((value or "").casefold().split())

    @staticmethod
    def _resume_context(profile: CandidateProfile) -> str:
        details = [profile.resume_summary or ""]
        details.extend(profile.skills + profile.frameworks + profile.databases + profile.interviewable_technologies + profile.experience)
        details.extend(f"{project.name}: {project.description}" for project in profile.projects)
        return "; ".join(part for part in details if part) or "Not provided"

    @staticmethod
    def _coerce_result(model_type, result):
        if isinstance(result, model_type):
            return result
        return model_type.model_validate(result)

    @staticmethod
    def _replace_warning(plan: InterviewPlan, index: int, warning: Optional[str]) -> None:
        prefix = f"Question {index + 1}:"
        plan.repair_warnings = [item for item in plan.repair_warnings if not item.startswith(prefix)]
        if warning:
            plan.repair_warnings.append(f"{prefix} {warning}")

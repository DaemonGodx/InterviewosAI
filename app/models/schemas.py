from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone

class Project(BaseModel):
    name: str
    description: str
    technologies: List[str]
    responsibilities: Optional[str] = None

class CandidateProfile(BaseModel):
    name: Optional[str] = "Unknown"
    education: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    databases: List[str] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    experience: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    resume_summary: Optional[str] = Field(None, description="A brief summary of the candidate's background")
    inferred_strengths: List[str] = Field(default_factory=list, description="Strengths inferred from the resume")
    potential_weaknesses: List[str] = Field(default_factory=list, description="Potential weak or unclear areas in the resume")
    interviewable_technologies: List[str] = Field(default_factory=list, description="Notable technologies or projects that can be questioned during the interview")
    estimated_years_experience: Optional[float] = Field(None, description="Estimated years of professional experience from the resume")
    recommended_experience_level: Optional[str] = Field(None, description="Recommended experience level string (e.g. 'Student', 'Fresher', '0-2 years', '2-5 years', '5+ years')")

class InterviewConfig(BaseModel):
    target_role: str
    experience_level: str
    interview_type: Literal["Technical", "Behavioral", "Mixed"] = "Mixed"
    question_count: int = Field(default=5, ge=1)
    starting_difficulty: Literal["Easy", "Medium", "Hard"] = "Medium"
    interview_mode: Literal["Text", "Voice"] = "Text"

BehavioralQuestionCategory = Literal["HR_MOTIVATION", "EXPERIENCE_BASED", "SITUATIONAL"]

class InterviewQuestion(BaseModel):
    question: str = Field(description="The actual question text to ask the candidate.")
    topic: str = Field(description="The topic this question covers.")
    primary_competency: Optional[str] = Field(None, description="The primary ability or evidence this question evaluates.")
    difficulty: str = Field(description="The difficulty level of this question.")
    question_type: Literal["Technical", "Behavioral"] = Field(description="Type of question: Technical or Behavioral.")
    behavioral_category: Optional[BehavioralQuestionCategory] = Field(
        default=None,
        description="Behavioral subtype: HR/motivation, experience-based behavior, or situational/hypothetical. Must be set for behavioral plan questions.",
    )
    expected_concepts: List[str] = Field(description="List of concepts expected in a strong answer.")

class QuestionSemanticValidation(BaseModel):
    matches_intended_category: bool = Field(description="Whether the question's main evaluation objective matches the intended category.")
    matches_intended_competency: bool = Field(default=False, description="Whether the actual assessment objective substantially matches the role-aware competency assigned to this slot.")
    matches_intended_behavioral_category: bool = Field(default=False, description="Whether a Behavioral question's actual intent matches its assigned HR/motivation, experience-based, or situational subtype.")
    primary_evaluation_objective: str = Field(description="The ability or evidence the candidate's answer is primarily expected to demonstrate.")
    reason: str = Field(description="Why the primary objective does or does not match the intended category.")
    assessed_competency: str = Field(default="", description="A short normalized name for the underlying competency actually assessed, independent of wording.")
    is_semantic_duplicate: bool = Field(default=False, description="Whether this question substantially assesses the same underlying competency as one of the supplied plan questions.")
    duplicate_reason: Optional[str] = Field(default=None, description="Why this question duplicates an existing question, if applicable.")

class GeneratedQuestionReview(BaseModel):
    """Question plus its semantic review, returned in one planner round-trip."""
    question: InterviewQuestion
    semantic_validation: QuestionSemanticValidation

class AdaptiveDecision(BaseModel):
    """Validated next-step decision produced alongside answer analysis."""
    action: Literal["ASK_FOLLOWUP", "ADVANCE", "FINISH"]
    rationale: str = Field(description="Brief internal reason based on the answer and collected evidence.")
    missing_concepts: List[str] = Field(default_factory=list, description="Specific unresolved concepts or evidence gaps, if any.")
    competency_sufficient: bool = Field(description="Whether the current planned competency has sufficient evidence to stop probing.")
    follow_up_question: Optional[InterviewQuestion] = Field(default=None, description="A focused follow-up question, required only for ASK_FOLLOWUP.")
    source_plan_index: Optional[int] = Field(default=None, ge=0, description="Stable zero-based planned-question index this decision and follow-up belong to.")
    follow_up_is_duplicate: bool = Field(description="Whether the proposed follow-up semantically repeats an already asked question.")

    @classmethod
    def supported_actions(cls):
        return ("ASK_FOLLOWUP", "ADVANCE", "FINISH")

class InterviewPlan(BaseModel):
    questions: List[InterviewQuestion] = Field(description="The complete ordered interview plan.")
    semantic_validations: List[QuestionSemanticValidation] = Field(default_factory=list, description="Semantic category validation for each question, in question order.")
    repair_warnings: List[str] = Field(default_factory=list, description="Questions replaced with safe fallbacks after bounded repair attempts.")

class AnswerAnalysis(BaseModel):
    correctness_score: int = Field(ge=0, le=10)
    depth_score: int = Field(ge=0, le=10)
    communication_score: int = Field(ge=0, le=10)
    feedback: str
    is_satisfactory: bool
    status: Literal["success", "failed"] = Field(default="success", description="Whether analysis succeeded or failed.")
    adaptive_decision: Optional[Dict[str, Any]] = Field(default=None, description="Structured adaptive recommendation returned with the answer analysis; validated separately so invalid decisions do not erase valid scoring evidence.")
    answer_classification: Literal["blank", "admitted_unknown", "irrelevant", "partially_correct", "substantive", "analysis_failed"] = Field(
        default="substantive", description="Classification of the response content, or analysis_failed when scoring could not be completed."
    )


class TurnIntent(BaseModel):
    """Validated classification of one candidate utterance."""
    intent: Literal[
        "ANSWER", "ADMITTED_UNKNOWN", "REPEAT", "CLARIFY", "HINT",
        "SKIP", "CORRECTION", "OFF_TOPIC", "AMBIGUOUS",
    ]
    confidence: float = Field(ge=0, le=1)
    requested_action: Optional[Literal["EASIER", "FOLLOW_UP", "HINT", "CLARIFY", "REPEAT", "SKIP"]] = Field(
        default=None,
        description="Explicit requested next action, kept separate from whether the utterance admits not knowing.",
    )

class EvidenceSummary(BaseModel):
    demonstrated_competence: List[str] = Field(default_factory=list)
    partial_evidence: List[str] = Field(default_factory=list)
    insufficient_evidence: List[str] = Field(default_factory=list)
    demonstrated_knowledge_gaps: List[str] = Field(default_factory=list)

class EvaluationReport(BaseModel):
    overall_score: Optional[float] = Field(None, ge=0, le=100)
    technical_score: Optional[float] = Field(None, ge=0, le=10)
    problem_solving_score: Optional[float] = Field(None, ge=0, le=10)
    communication_score: Optional[float] = Field(None, ge=0, le=10)
    behavioral_score: Optional[float] = Field(None, ge=0, le=10)
    evaluation_confidence: str = Field(description="Confidence based on usable successfully analyzed answers and competency breadth: Very Low, Low, Moderate, High")
    resume_claims: List[str] = Field(description="Strengths claimed on the resume")
    interview_demonstrated_strengths: List[str] = Field(description="Strengths actually demonstrated during the interview")
    weaknesses: List[str] = Field(description="Areas to improve demonstrated during the interview")
    technical_gaps: List[str]
    recommended_topics: List[str]
    role_readiness: str
    final_feedback: str
    evidence_summary: EvidenceSummary = Field(default_factory=EvidenceSummary)
    readiness_basis: Optional[str] = None

class InterviewState(BaseModel):
    session_id: str
    config: InterviewConfig
    profile: CandidateProfile
    interview_plan: List[InterviewQuestion] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    answers: List[str] = Field(default_factory=list)
    analyses: List[AnswerAnalysis] = Field(default_factory=list)
    # Decision history is intentionally schema-light so sessions saved by older
    # releases (whose action vocabulary differed) remain readable.
    decisions: List[dict] = Field(default_factory=list)
    topics_covered: List[str] = Field(default_factory=list)
    current_difficulty: str = "Medium"
    current_question: Optional[str] = None
    current_question_type: Optional[str] = None
    current_behavioral_category: Optional[BehavioralQuestionCategory] = None
    current_question_topic: Optional[str] = None
    current_question_difficulty: Optional[str] = None
    current_primary_competency: Optional[str] = None
    current_expected_concepts: List[str] = Field(default_factory=list)
    current_plan_index: Optional[int] = None
    current_follow_up_for_plan_index: Optional[int] = None
    question_metadata: List[dict] = Field(default_factory=list)
    plan_cursor: int = 0
    plan_execution_order: List[int] = Field(default_factory=list)
    pending_adaptive_question: Optional[dict] = None
    next_difficulty_override: Optional[str] = None
    follow_up_counts: Dict[str, int] = Field(default_factory=dict)
    adaptation_history: List[dict] = Field(default_factory=list)
    remaining_question_count: Optional[int] = None
    current_answer: Optional[str] = None
    current_input: Optional[str] = None
    classified_utterance: Optional[str] = None
    current_turn_intent: Optional[dict] = None
    interviewer_response: Optional[str] = None
    conversation_history: List[dict] = Field(default_factory=list)
    consecutive_ambiguous_turns: int = 0
    status: str = "IN_PROGRESS"
    evaluation: Optional[EvaluationReport] = None
    evaluation_error: Optional[str] = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

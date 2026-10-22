"""LangGraph state normalization.

The graph boundary uses JSON-compatible dictionaries throughout. Pydantic models
validate nested domain values when entering and leaving the graph; nodes then
operate on the same dictionary representation produced by the CLI and SQLite.
"""

from typing import Any, Dict, List, Optional, TypedDict

from app.models.schemas import (
    AnswerAnalysis,
    CandidateProfile,
    EvaluationReport,
    InterviewConfig,
    InterviewQuestion,
)


class GraphState(TypedDict, total=False):
    session_id: str
    config: Dict[str, Any]
    profile: Dict[str, Any]
    interview_plan: List[Dict[str, Any]]
    questions: List[str]
    answers: List[str]
    analyses: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]]
    topics_covered: List[str]
    current_difficulty: str
    current_question: Optional[str]
    current_question_type: Optional[str]
    current_behavioral_category: Optional[str]
    current_question_topic: Optional[str]
    current_question_difficulty: Optional[str]
    current_primary_competency: Optional[str]
    current_expected_concepts: List[str]
    current_plan_index: Optional[int]
    current_follow_up_for_plan_index: Optional[int]
    question_metadata: List[dict]
    plan_cursor: int
    plan_execution_order: List[int]
    pending_adaptive_question: Optional[dict]
    next_difficulty_override: Optional[str]
    follow_up_counts: Dict[str, int]
    adaptation_history: List[dict]
    remaining_question_count: Optional[int]
    current_answer: Optional[str]
    current_input: Optional[str]
    classified_utterance: Optional[str]
    current_turn_intent: Optional[Dict[str, Any]]
    interviewer_response: Optional[str]
    conversation_history: List[dict]
    consecutive_ambiguous_turns: int
    status: str
    evaluation: Optional[Dict[str, Any]]
    evaluation_error: Optional[str]


def _dump(model_type, value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return model_type.model_validate(value).model_dump(mode="json")


def normalize_graph_state(state: dict) -> GraphState:
    """Validate known state models and return JSON-compatible dictionaries."""
    if hasattr(state, "model_dump"):
        state = state.model_dump(mode="json")
    result = dict(state)
    if "config" in result:
        result["config"] = _dump(InterviewConfig, result["config"])
    if "profile" in result:
        result["profile"] = _dump(CandidateProfile, result["profile"])
    for key, model_type in (
        ("interview_plan", InterviewQuestion),
        ("analyses", AnswerAnalysis),
    ):
        if key in result:
            result[key] = [_dump(model_type, item) for item in result[key]]
    if result.get("evaluation") is not None:
        result["evaluation"] = _dump(EvaluationReport, result["evaluation"])

    # These are schema-free state collections but still normalize nested models,
    # enums, and datetimes into values safe for JSON and LangGraph checkpoints.
    for key in ("question_metadata", "adaptation_history", "pending_adaptive_question", "current_turn_intent", "conversation_history"):
        value = result.get(key)
        if value is not None:
            result[key] = _json_compatible(value)
    return _json_compatible(result)  # type: ignore[return-value]


def _json_compatible(value):
    from datetime import date, datetime
    from enum import Enum

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value

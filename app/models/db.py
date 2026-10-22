import json
from datetime import date, datetime, timezone
from enum import Enum
from collections.abc import Mapping
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from app.models.schemas import (
    AnswerAnalysis, CandidateProfile, EvaluationReport,
    InterviewConfig, InterviewQuestion,
)

Base = declarative_base()


def utc_now_naive():
    """SQLite DateTime columns use naive UTC for compatibility with old rows."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _as_mapping(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value) if isinstance(value, Mapping) else {}


def _json_default(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def _json_dump(value):
    return json.dumps(value, default=_json_default)


def _parse_timestamp(value):
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if result.tzinfo is not None:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result


def _validated_or_preserved(model_type, value, warnings, label):
    try:
        return model_type.model_validate(value).model_dump(mode="json")
    except Exception as error:
        warnings.append(f"Legacy {label} could not be normalized: {error}")
        return value

class SessionRecord(Base):
    __tablename__ = 'interview_sessions'

    session_id = Column(String, primary_key=True)
    candidate_name = Column(String, nullable=True)
    target_role = Column(String, nullable=False)
    experience_level = Column(String, nullable=False)
    profile_json = Column(Text, nullable=False)
    config_json = Column(Text, nullable=False)
    interview_data_json = Column(Text, nullable=False) # Includes questions, answers, analyses, decisions
    evaluation_json = Column(Text, nullable=True)
    started_at = Column(DateTime, default=utc_now_naive)
    completed_at = Column(DateTime, nullable=True)

class Database:
    def __init__(self, db_path="interviewos.db"):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_session(self, state_dict: dict):
        state_dict = _as_mapping(state_dict)
        config = _as_mapping(state_dict.get('config', {}))
        profile = _as_mapping(state_dict.get('profile', {}))
        with self.Session() as session:
            record = session.query(SessionRecord).filter_by(session_id=state_dict['session_id']).first()
            if not record:
                record = SessionRecord(session_id=state_dict['session_id'])
                session.add(record)
            
            record.candidate_name = profile.get('name', 'Unknown')
            record.target_role = config.get('target_role', '')
            record.experience_level = config.get('experience_level', '')
            record.profile_json = _json_dump(profile)
            record.config_json = _json_dump(config)
            started_at = _parse_timestamp(state_dict.get('started_at'))
            if started_at is not None:
                record.started_at = started_at
            
            interview_data = {
                'interview_plan': state_dict.get('interview_plan', []),
                'questions': state_dict.get('questions', []),
                'answers': state_dict.get('answers', []),
                'analyses': state_dict.get('analyses', []),
                'decisions': state_dict.get('decisions', []),
                'topics_covered': state_dict.get('topics_covered', []),
                'question_metadata': state_dict.get('question_metadata', []),
                'adaptation_history': state_dict.get('adaptation_history', []),
                'follow_up_counts': state_dict.get('follow_up_counts', {}),
                'plan_execution_order': state_dict.get('plan_execution_order', []),
                'plan_cursor': state_dict.get('plan_cursor', 0),
                'remaining_question_count': state_dict.get('remaining_question_count'),
                'conversation_history': state_dict.get('conversation_history', []),
                'consecutive_ambiguous_turns': state_dict.get('consecutive_ambiguous_turns', 0),
            }
            record.interview_data_json = _json_dump(interview_data)
            
            evaluation = state_dict.get('evaluation')
            if evaluation:
                record.evaluation_json = _json_dump(evaluation)
                record.completed_at = utc_now_naive()
                
            session.commit()

    def get_all_sessions(self):
        with self.Session() as session:
            records = session.query(SessionRecord).order_by(SessionRecord.started_at.desc()).all()
            return records

    def get_session(self, session_id: str):
        with self.Session() as session:
            return session.query(SessionRecord).filter_by(session_id=session_id).first()

    def load_session(self, session_id: str):
        """Load a normalized session; retain older raw values with warnings."""
        record = self.get_session(session_id)
        if record is None:
            return None
        warnings = []
        try:
            config_data = json.loads(record.config_json or "{}")
        except (TypeError, json.JSONDecodeError):
            config_data = {}
            warnings.append("Saved config JSON was malformed.")
        try:
            profile_data = json.loads(record.profile_json or "{}")
        except (TypeError, json.JSONDecodeError):
            profile_data = {}
            warnings.append("Saved profile JSON was malformed.")
        try:
            interview_data = json.loads(record.interview_data_json or "{}")
        except (TypeError, json.JSONDecodeError):
            interview_data = {}
            warnings.append("Saved interview history JSON was malformed.")
        if not isinstance(interview_data, dict):
            interview_data = {}
            warnings.append("Saved interview history must be a JSON object.")
        try:
            evaluation_data = json.loads(record.evaluation_json) if record.evaluation_json else None
        except (TypeError, json.JSONDecodeError):
            evaluation_data = None
            warnings.append("Saved evaluation JSON was malformed.")

        config_data = _validated_or_preserved(InterviewConfig, config_data, warnings, "config")
        profile_data = _validated_or_preserved(CandidateProfile, profile_data, warnings, "profile")
        for key, model_type in (
            ("interview_plan", InterviewQuestion),
            ("analyses", AnswerAnalysis),
        ):
            values = interview_data.get(key, [])
            normalized = []
            for index, item in enumerate(values):
                normalized.append(_validated_or_preserved(model_type, item, warnings, f"{key}[{index}]"))
            interview_data[key] = normalized
        # Decision history is retained verbatim because earlier releases used
        # different action names and fields. Current decisions are validated at
        # the adaptive graph node before they can affect routing.
        if not isinstance(interview_data.get("decisions", []), list):
            warnings.append("Saved decision history was not a list and was reset.")
            interview_data["decisions"] = []
        if evaluation_data is not None:
            evaluation_data = _validated_or_preserved(EvaluationReport, evaluation_data, warnings, "evaluation")

        # Fields added after initial releases are additive defaults on read.
        for key, default in (
            ("questions", []), ("answers", []), ("analyses", []), ("decisions", []),
            ("topics_covered", []), ("interview_plan", []), ("question_metadata", []),
            ("adaptation_history", []), ("follow_up_counts", {}),
            ("plan_execution_order", []), ("plan_cursor", 0), ("remaining_question_count", None),
            ("conversation_history", []), ("consecutive_ambiguous_turns", 0),
        ):
            interview_data.setdefault(key, default)

        started_at = record.started_at
        if started_at is not None and started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        elif started_at is not None:
            started_at = started_at.astimezone(timezone.utc)
        completed_at = record.completed_at
        if completed_at is not None and completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=timezone.utc)
        elif completed_at is not None:
            completed_at = completed_at.astimezone(timezone.utc)
        return {
            "session_id": record.session_id,
            "candidate_name": record.candidate_name,
            "target_role": record.target_role,
            "experience_level": record.experience_level,
            "started_at": started_at.isoformat() if started_at else None,
            "completed_at": completed_at.isoformat() if completed_at else None,
            "config": config_data,
            "profile": profile_data,
            "interview_data": interview_data,
            "evaluation": evaluation_data,
            "compatibility_warnings": warnings,
        }

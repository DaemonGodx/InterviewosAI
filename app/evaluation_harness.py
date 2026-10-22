"""Offline, fixture-driven evaluation of InterviewOS conversational paths.

This module is deliberately outside the production interview flow. Its fake
model supplies known structured outputs while the compiled production graph,
nodes, schemas, and adaptive policy perform the work under evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

from pydantic import BaseModel, Field, ValidationError


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "tests" / "evaluations" / "datasets" / "conversation_cases_v1.json"
RUBRIC_PATH = ROOT / "tests" / "evaluations" / "rubric_v1.json"


class Case(BaseModel):
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    dimension: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    question: dict[str, Any]
    utterance: str = Field(min_length=1)
    fixture: dict[str, Any]
    expected: dict[str, Any]


def load_cases(dataset_path: Path = DATASET_PATH) -> tuple[str, list[Case]]:
    """Load and validate a versioned JSON dataset, with actionable errors."""
    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read evaluation dataset {dataset_path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise ValueError("Dataset must be an object with a 'cases' array.")
    version = raw.get("dataset_version")
    if not isinstance(version, str) or not version:
        raise ValueError("Dataset must define a non-empty 'dataset_version'.")
    cases: list[Case] = []
    seen: set[str] = set()
    for index, item in enumerate(raw["cases"]):
        try:
            case = Case.model_validate(item)
        except ValidationError as exc:
            raise ValueError(f"Invalid evaluation case at cases[{index}]: {exc}") from exc
        if case.id in seen:
            raise ValueError(f"Duplicate evaluation case id {case.id!r} at cases[{index}].")
        seen.add(case.id)
        if case.expected.get("intent") not in {
            "ANSWER", "ADMITTED_UNKNOWN", "REPEAT", "CLARIFY", "HINT", "SKIP",
            "CORRECTION", "OFF_TOPIC", "AMBIGUOUS",
        }:
            raise ValueError(f"Case {case.id!r} has an unsupported expected intent.")
        if not isinstance(case.question.get("question"), str) or not case.question["question"].strip():
            raise ValueError(f"Case {case.id!r} must define a non-empty question.question string.")
        for key in ("scored", "same_question", "preserve_progress"):
            if key not in case.expected or not isinstance(case.expected[key], bool):
                raise ValueError(f"Case {case.id!r} expected.{key} must be a boolean.")
        adaptive_action = case.expected.get("adaptive_action")
        if adaptive_action is not None and adaptive_action not in {"ASK_FOLLOWUP", "ADVANCE", "FINISH"}:
            raise ValueError(f"Case {case.id!r} has an unsupported expected adaptive_action.")
        forbidden = case.expected.get("forbidden_answer_terms", [])
        if not isinstance(forbidden, list) or any(not isinstance(term, str) for term in forbidden):
            raise ValueError(f"Case {case.id!r} expected.forbidden_answer_terms must be an array of strings.")
        if case.fixture.get("intent") not in {
            "ANSWER", "ADMITTED_UNKNOWN", "REPEAT", "CLARIFY", "HINT", "SKIP",
            "CORRECTION", "OFF_TOPIC", "AMBIGUOUS",
        }:
            raise ValueError(f"Case {case.id!r} fixture must include a supported intent.")
        cases.append(case)
    if not cases:
        raise ValueError("Dataset contains no evaluation cases.")
    return version, cases


def load_rubric(rubric_path: Path = RUBRIC_PATH) -> dict[str, Any]:
    try:
        rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read evaluation rubric {rubric_path}: {exc}") from exc
    if not isinstance(rubric, dict) or not rubric.get("rubric_version"):
        raise ValueError("Rubric must be an object with a non-empty 'rubric_version'.")
    return rubric


class _FakePrompt:
    def __or__(self, model):
        return _FakeChain(model)


class _FakeModel:
    def __init__(self, fixture: dict[str, Any]):
        self.fixture = fixture

    def with_structured_output(self, schema):
        return _FakeStructuredModel(self, schema)


class _FakeStructuredModel:
    def __init__(self, model: _FakeModel, schema):
        self.model, self.schema = model, schema


class _FakeChain:
    def __init__(self, model):
        self.model = model

    def invoke(self, _inputs):
        fixture = self.model.model.fixture if isinstance(self.model, _FakeStructuredModel) else self.model.fixture
        schema = self.model.schema if isinstance(self.model, _FakeStructuredModel) else None
        if schema is None:
            return fixture.get("response", "Please focus on the question and share your approach.")
        if schema.__name__ == "TurnIntent":
            return {"intent": fixture["intent"], "confidence": fixture.get("confidence", 0.99)}
        if schema.__name__ == "AnswerAnalysis":
            return fixture["analysis"]
        raise AssertionError(f"Unexpected model schema requested by production path: {schema.__name__}")


def _initial_state(case: Case) -> dict[str, Any]:
    q = case.question
    question = q["question"]
    first = {
        "question": question,
        "topic": q.get("topic", "General"),
        "primary_competency": q.get("primary_competency", q.get("topic", "General")),
        "difficulty": q.get("difficulty", "Medium"),
        "question_type": q.get("question_type", "Technical"),
        "expected_concepts": q.get("expected_concepts", []),
    }
    future = {
        "question": "What evidence would you use to verify the outcome?",
        "topic": first["topic"],
        "primary_competency": first["primary_competency"],
        "difficulty": first["difficulty"],
        "question_type": first["question_type"],
        "expected_concepts": first["expected_concepts"],
    }
    return {
        "session_id": f"evaluation-{case.id}",
        "config": {"target_role": "Software Engineer", "experience_level": "Mid-level", "question_count": 3, "interview_type": "Technical"},
        "profile": {}, "interview_plan": [first, future, {**future, "question": "How would you handle a related edge case?"}],
        "questions": [question], "answers": [], "analyses": [], "decisions": [],
        "topics_covered": [first["topic"]],
        "question_metadata": [{**first, "plan_index": 0, "source_plan_index": 0, "is_follow_up": False}],
        "current_question": question, "current_question_type": first["question_type"],
        "current_question_topic": first["topic"], "current_question_difficulty": first["difficulty"],
        "current_behavioral_category": q.get("behavioral_category"),
        "current_primary_competency": first["primary_competency"],
        "current_expected_concepts": first["expected_concepts"],
        "current_plan_index": 0, "current_follow_up_for_plan_index": None,
        "plan_cursor": 1, "plan_execution_order": [0, 1, 2], "remaining_question_count": 2,
        "follow_up_counts": {}, "adaptation_history": [], "conversation_history": case.fixture.get("prior_conversation", []),
        "consecutive_ambiguous_turns": case.fixture.get("prior_ambiguous_turns", 0), "status": "IN_PROGRESS",
    }


def _analysis_fixture(case: Case) -> dict[str, Any]:
    adaptive = case.fixture.get("adaptive", {})
    action = adaptive.get("action", "ADVANCE")
    follow_up = adaptive.get("follow_up_question")
    return {
        "correctness_score": case.fixture.get("correctness_score", 6),
        "depth_score": case.fixture.get("depth_score", 5),
        "communication_score": 6,
        "feedback": "Fixture assessment used for deterministic path verification.",
        "is_satisfactory": action != "ASK_FOLLOWUP",
        "status": "success",
        "answer_classification": case.fixture.get("answer_classification", "substantive"),
        "adaptive_decision": {
            "action": action,
            "rationale": adaptive.get("rationale", "Fixture decision for policy-path evaluation."),
            "missing_concepts": adaptive.get("missing_concepts", []),
            "competency_sufficient": adaptive.get("competency_sufficient", action != "ASK_FOLLOWUP"),
            "follow_up_question": follow_up,
            "source_plan_index": 0,
            "follow_up_is_duplicate": adaptive.get("follow_up_is_duplicate", False),
        },
    }


def _run_case(case: Case) -> dict[str, Any]:
    # Import only for explicit evaluation; replacing the model and prompt factory
    # prevents all network access while retaining the production graph and nodes.
    from app.engine.graph import InterviewEngine
    from app.engine import nodes

    state = _initial_state(case)
    fixture = {**case.fixture, "analysis": _analysis_fixture(case)}
    model = _FakeModel(fixture)
    with patch.object(nodes, "llm", model), patch.object(nodes.ChatPromptTemplate, "from_messages", return_value=_FakePrompt()):
        result = InterviewEngine().process_answer(state, case.utterance)

    expected = case.expected
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, observed: Any, reason: str):
        checks.append({"name": name, "status": "pass" if condition else "fail", "observed": observed, "reason": reason})

    actual_intent = (result.get("current_turn_intent") or {}).get("intent")
    check("production_route_intent", actual_intent == expected["intent"], actual_intent,
          "The production graph routed the deterministic intent fixture through its expected branch.")
    is_scored = len(result.get("answers", [])) == 1 and len(result.get("analyses", [])) == 1
    check("scoring_path", is_scored == expected.get("scored", False),
          {"answers": len(result.get("answers", [])), "analyses": len(result.get("analyses", []))},
          "Checks whether production graph created the expected scored-answer record.")
    same_question = result.get("current_question") == state["current_question"]
    check("active_question", same_question == expected.get("same_question", False), result.get("current_question"),
          "Checks whether the active question remained unchanged as required by the case.")
    same_progress = (
        result.get("plan_cursor") == state["plan_cursor"]
        and result.get("follow_up_counts", {}) == state["follow_up_counts"]
        and len(result.get("questions", [])) == len(state["questions"])
    )
    progress_expected = expected.get("preserve_progress", False)
    check("progress_counters", same_progress == progress_expected,
          {"plan_cursor": result.get("plan_cursor"), "follow_up_counts": result.get("follow_up_counts"), "questions": len(result.get("questions", []))},
          "Checks question history, plan cursor, and follow-up counts.")
    if actual_intent not in {"ANSWER", "ADMITTED_UNKNOWN", "SKIP"}:
        check("separate_conversation_history", bool(result.get("conversation_history")) and not result.get("answers") and not result.get("analyses"),
              {"conversation_history": len(result.get("conversation_history", [])), "answers": len(result.get("answers", [])), "analyses": len(result.get("analyses", []))},
              "Conversation-only turns must stay out of aligned scored answer and analysis arrays.")
    if actual_intent == "SKIP":
        analysis = (result.get("analyses") or [{}])[-1]
        check("skip_no_evidence", result.get("answers", [None])[-1] == "" and analysis.get("answer_classification") == "blank" and analysis.get("correctness_score") == 0,
              {"answer": result.get("answers", [None])[-1], "analysis": analysis},
              "Skip follows the production blank/no-evidence analysis path with zero candidate points.")
        check("skip_advances", result.get("plan_cursor", 0) > state["plan_cursor"], result.get("plan_cursor"),
              "Skip consumes the current planned item through the existing progression path.")
    if actual_intent == "ADMITTED_UNKNOWN" and result.get("analyses"):
        check("unknown_compatibility", result["analyses"][-1].get("answer_classification") == "admitted_unknown",
              result["analyses"][-1].get("answer_classification"),
              "Production analysis preserves admitted_unknown classification.")
    if case.fixture.get("prior_conversation"):
        prior_utterances = {item.get("utterance") for item in case.fixture["prior_conversation"]}
        check("conversation_not_answer_evidence", not (prior_utterances & set(result.get("answers", []))) and len(result.get("analyses", [])) == 1,
              {"prior_conversation_count": len(case.fixture["prior_conversation"]), "answers": result.get("answers"), "analyses": len(result.get("analyses", []))},
              "Earlier conversational utterances remain outside scored answer evidence when a later substantive answer is analyzed.")
    if actual_intent == "REPEAT":
        response_ok = result.get("interviewer_response") == state["current_question"]
        check("repeat_exact_question", response_ok, result.get("interviewer_response"),
              "Repeat response must reproduce the exact active question.")
    if actual_intent == "AMBIGUOUS":
        response = (result.get("interviewer_response") or "").casefold()
        required = ["answer", "hint", "skip"]
        is_bounded_fallback = result.get("consecutive_ambiguous_turns", 0) >= 2
        check("ambiguous_bounded_fallback", not is_bounded_fallback or all(word in response for word in required),
              result.get("interviewer_response"),
              "At the configured ambiguity limit, the response must offer answer, hint, or skip.")
    if actual_intent in {"CORRECTION", "OFF_TOPIC", "CLARIFY"}:
        check("conversation_response_present", bool((result.get("interviewer_response") or "").strip()),
              result.get("interviewer_response"), "The conversation-only path must return a response to the CLI.")
    if actual_intent == "HINT":
        response = result.get("interviewer_response") or ""
        forbidden = case.expected.get("forbidden_answer_terms", [])
        leaked = [term for term in forbidden if term.casefold() in response.casefold()]
        checks.append({"name": "hint_answer_leakage", "status": "pass" if not leaked else "fail", "observed": response[:240],
                       "reason": "Deterministic forbidden-term screen; passing does not establish semantic hint quality." if not leaked else f"Found forbidden answer terms: {leaked}"})
        checks.append({"name": "hint_semantic_usefulness", "status": "inconclusive", "observed": response[:240],
                       "reason": "The response is supplied by an offline fixture, so usefulness cannot be judged as live model behavior."})
    if expected.get("adaptive_action"):
        decisions = result.get("decisions", [])
        actual_action = decisions[-1].get("action") if decisions else None
        check("adaptive_policy_action", actual_action == expected["adaptive_action"], actual_action,
              "The production adaptive decision node validated and applied the fixture recommendation.")
        if expected["adaptive_action"] == "ASK_FOLLOWUP":
            followup_meta = (result.get("question_metadata") or [{}])[-1]
            check("adaptive_followup_staged", bool(followup_meta.get("is_follow_up")) and followup_meta.get("follow_up_for_plan_index") == 0,
                  followup_meta, "A valid follow-up must be generated and remain linked to its source plan item.")
        checks.append({"name": "adaptive_semantic_relevance", "status": "inconclusive", "observed": expected.get("adaptive_action"),
                       "reason": "Fixture recommendations exercise production policy validation, but cannot establish live semantic relevance."})
    # Every case reports the limit of fixture-driven intent evaluation explicitly.
    checks.append({"name": "natural_language_intent_accuracy", "status": "inconclusive", "observed": case.utterance[:160],
                   "reason": "Intent is an oracle fixture in offline mode; no live classifier judgment is made."})
    return {
        "id": case.id, "category": case.category, "dimension": case.dimension,
        "scenario": case.scenario,
        "observed": {"intent": actual_intent, "response": (result.get("interviewer_response") or "")[:300],
                     "answer_count": len(result.get("answers", [])), "analysis_count": len(result.get("analyses", [])),
                     "current_question": result.get("current_question"), "plan_cursor": result.get("plan_cursor"),
                     "decisions": result.get("decisions", [])[-1:]},
        "checks": checks,
        # Case status summarizes deterministic production-contract checks only;
        # semantic checks remain individually visible as inconclusive.
        "status": "fail" if any(c["status"] == "fail" for c in checks) else ("pass" if any(c["status"] == "pass" for c in checks) else "inconclusive"),
    }


def evaluate(case_ids: list[str] | None = None, dataset_path: Path = DATASET_PATH, rubric_path: Path = RUBRIC_PATH) -> dict[str, Any]:
    dataset_version, cases = load_cases(dataset_path)
    rubric = load_rubric(rubric_path)
    if case_ids:
        requested = set(case_ids)
        known = {case.id for case in cases}
        missing = sorted(requested - known)
        if missing:
            raise ValueError(f"Unknown evaluation case id(s): {', '.join(missing)}")
        cases = [case for case in cases if case.id in requested]
    results = [_run_case(case) for case in cases]
    counts = Counter(row["status"] for row in results)
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0, "inconclusive": 0})
    categories: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0, "inconclusive": 0})
    check_counts = Counter()
    for row in results:
        categories[row["category"]][row["status"]] += 1
        for check in row["checks"]:
            check_counts[check["status"]] += 1
            grouped[row["dimension"]][check["status"]] += 1
    return {
        "report_version": "1.0", "mode": "offline", "dataset_version": dataset_version,
        "rubric_version": rubric["rubric_version"], "total_cases": len(results),
        "summary": {key: counts.get(key, 0) for key in ("pass", "fail", "inconclusive")},
        "check_summary": {key: check_counts.get(key, 0) for key in ("pass", "fail", "inconclusive")},
        "case_status_basis": "Case pass/fail summarizes deterministic production-contract checks. Semantic limitations remain visible as per-check inconclusive results and are not counted as passing semantic judgments.",
        "by_dimension": dict(grouped), "by_category": dict(categories),
        "skipped_tests": [],
        "unavailable_capabilities": [
            "Live-model semantic evaluation is not implemented; production has no independent judge seam.",
            "Natural-language intent accuracy and model-generated prose quality are not established by offline fixtures.",
        ],
        "limitations": [
            "Intent wording accuracy is inconclusive offline because fixture intents are supplied to the production graph.",
            "Semantic response usefulness and adaptive relevance are inconclusive when they depend on model-generated prose.",
            "Hint leakage checks use the case's deterministic forbidden-term list and are not a semantic proof.",
        ],
        "results": results,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"InterviewOS evaluation ({report['mode']}) | dataset {report['dataset_version']} | rubric {report['rubric_version']}")
    print(f"Contract cases: {report['total_cases']} | pass {report['summary']['pass']} | fail {report['summary']['fail']} | inconclusive {report['summary']['inconclusive']}")
    print(f"Checks: pass {report['check_summary']['pass']} | fail {report['check_summary']['fail']} | inconclusive {report['check_summary']['inconclusive']}")
    for dimension, counts in report["by_dimension"].items():
        print(f"  {dimension}: pass {counts['pass']}, fail {counts['fail']}, inconclusive {counts['inconclusive']}")
    for row in report["results"]:
        print(f"[{row['status'].upper()}] {row['id']} ({row['category']}) - {row['scenario']}")
        for check in row["checks"]:
            if check["status"] != "pass":
                print(f"    {check['status']}: {check['name']} - {check['reason']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic offline InterviewOS behavior evaluations.")
    parser.add_argument("--case", action="append", dest="case_ids", help="Run a selected case ID (repeatable).")
    parser.add_argument("--json", type=Path, dest="json_path", help="Also write the full report as JSON to this path.")
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.case_ids)
    except ValueError as exc:
        parser.error(str(exc))
    _print_report(report)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"JSON report written to {args.json_path}")
    return 1 if report["summary"]["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())

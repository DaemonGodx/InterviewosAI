import json
from pathlib import Path

import pytest

from app import evaluation_harness as harness


def test_dataset_is_versioned_unique_and_covers_required_categories():
    version, cases = harness.load_cases()

    assert version == "1.0.0"
    assert len(cases) >= 20
    assert len({case.id for case in cases}) == len(cases)
    assert {"answer_analysis", "conversation", "skip", "adaptive_follow_up"} <= {case.category for case in cases}


def _patch_virtual_json(monkeypatch, path, content):
    original_read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *args, **kwargs: content if self == path else original_read_text(self, *args, **kwargs))


def test_dataset_loader_rejects_duplicate_ids_with_actionable_message(monkeypatch):
    path = Path("virtual-duplicate.json")
    _patch_virtual_json(monkeypatch, path, json.dumps({
        "dataset_version": "test",
        "cases": [
            {"id": "same", "category": "x", "dimension": "x", "scenario": "x", "question": {"question": "q"}, "utterance": "a", "fixture": {"intent": "ANSWER"}, "expected": {"intent": "ANSWER", "scored": True, "same_question": False, "preserve_progress": False}},
            {"id": "same", "category": "x", "dimension": "x", "scenario": "x", "question": {"question": "q"}, "utterance": "a", "fixture": {"intent": "ANSWER"}, "expected": {"intent": "ANSWER", "scored": True, "same_question": False, "preserve_progress": False}},
        ],
    }))

    with pytest.raises(ValueError, match="Duplicate evaluation case id 'same'"):
        harness.load_cases(path)


def test_dataset_loader_rejects_malformed_case_with_index(monkeypatch):
    path = Path("virtual-malformed.json")
    _patch_virtual_json(monkeypatch, path, json.dumps({"dataset_version": "test", "cases": [{"id": "bad"}]}))

    with pytest.raises(ValueError, match=r"cases\[0\]"):
        harness.load_cases(path)


@pytest.mark.parametrize("case_id,expected_intent,check_name", [
    ("repeat-08", "REPEAT", "repeat_exact_question"),
    ("skip-11", "SKIP", "skip_no_evidence"),
    ("hint-first-10", "HINT", "hint_answer_leakage"),
    ("admitted-unknown-06", "ADMITTED_UNKNOWN", "unknown_compatibility"),
    ("ambiguous-limit-15", "SKIP", "skip_no_evidence"),
    ("answer-partial-02", "ANSWER", "adaptive_followup_staged"),
])
def test_offline_cases_exercise_production_graph_with_fixture_model(case_id, expected_intent, check_name):
    report = harness.evaluate([case_id])
    case = report["results"][0]
    checks = {check["name"]: check for check in case["checks"]}

    assert report["mode"] == "offline"
    assert case["observed"]["intent"] == expected_intent
    assert checks[check_name]["status"] == "pass"
    assert case["status"] == "pass"


def test_evaluation_reports_failures_and_semantic_inconclusive_checks(monkeypatch):
    version, cases = harness.load_cases()
    raw = json.loads(harness.DATASET_PATH.read_text(encoding="utf-8"))
    raw["cases"][0]["expected"]["intent"] = "REPEAT"
    dataset_path = Path("virtual-changed.json")
    _patch_virtual_json(monkeypatch, dataset_path, json.dumps(raw))

    report = harness.evaluate([cases[0].id], dataset_path=dataset_path)

    assert report["dataset_version"] == version
    assert report["summary"]["fail"] == 1
    check_statuses = {check["status"] for check in report["results"][0]["checks"]}
    assert {"pass", "fail", "inconclusive"} <= check_statuses


def test_unknown_case_selection_is_rejected():
    with pytest.raises(ValueError, match="Unknown evaluation case id"):
        harness.evaluate(["not-a-case"])


def test_cli_prints_summary_and_writes_machine_readable_json(monkeypatch, capsys):
    written = {}
    monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: None)
    monkeypatch.setattr(Path, "write_text", lambda self, content, **kwargs: written.update(path=str(self), content=content))

    status = harness.main(["--case", "repeat-08", "--json", "virtual-output/report.json"])

    report = json.loads(written["content"])
    assert status == 0
    assert report["total_cases"] == 1
    assert report["results"][0]["id"] == "repeat-08"
    assert "Contract cases: 1" in capsys.readouterr().out

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_acceptance import EXPECTED, check
from mlca.paths import sha256


def fixture_exports(tmp_path):
    records = tmp_path / "records"
    root = tmp_path / "repo"
    records.mkdir()
    for study, task in EXPECTED:
        (records / (study + "_" + task + ".json")).write_text(
            json.dumps(
                {
                    "study": study,
                    "task": task,
                    "model_requests": 0,
                    "status": "failed",
                    "core_completed": False,
                    "not_achieved": 0,
                }
            )
        )
        export = root / "bio-bench" / study / "runs/reference/2026-10-14-final" / task
        project = export / "run-001"
        log = project / ".mlca/log/events.jsonl"
        log.parent.mkdir(parents=True)
        log.write_text(
            json.dumps(
                {
                    "type": "StepFailed",
                    "event_id": "interpret",
                    "payload": {"category": "timeout"},
                }
            )
            + "\n"
        )
        custom = project / "custom_instructions.md"
        custom.write_text(
            json.dumps({"event": "model", "outcome": "not achieved"}) + "\n"
        )
        with (export / "EXPORT_MANIFEST.csv").open("w") as f:
            writer = csv.DictWriter(f, ["path", "sha256"])
            writer.writeheader()
            for file in (log, custom):
                writer.writerow(
                    {"path": str(file.relative_to(export)), "sha256": sha256(file)}
                )
    return records, root


def test_failed_runs_derive_custom_outcomes_and_failure_category(tmp_path):
    records, root = fixture_exports(tmp_path)
    rows = check(records, root, tmp_path / "summary")
    assert len(rows) == 7
    assert all(row["not_achieved"] == 1 for row in rows)
    assert all(row["last_failure_category"] == "timeout" for row in rows)
    assert all(row["last_failure_event"] == "interpret" for row in rows)
    assert (tmp_path / "summary/summary.csv").exists()


def test_acceptance_refuses_modified_export(tmp_path):
    records, root = fixture_exports(tmp_path)
    next(root.rglob("events.jsonl")).write_text("{}\n")
    with pytest.raises(AssertionError):
        check(records, root, tmp_path / "summary")


def test_acceptance_refuses_model_calls(tmp_path):
    records, root = fixture_exports(tmp_path)
    record = next(records.glob("*.json"))
    row = json.loads(record.read_text())
    row["model_requests"] = 1
    record.write_text(json.dumps(row))
    with pytest.raises(AssertionError):
        check(records, root, tmp_path / "summary")


def test_acceptance_refuses_unlisted_file(tmp_path):
    records, root = fixture_exports(tmp_path)
    manifest = next(root.rglob("EXPORT_MANIFEST.csv"))
    (manifest.parent / "unexpected.txt").write_text("unexpected")
    with pytest.raises(AssertionError):
        check(records, root, tmp_path / "summary")

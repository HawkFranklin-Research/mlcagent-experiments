import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import run_batch as batch
from mlca.rails import load


def test_queue_limits_parallelism_and_large_studies():
    jobs = [{"id": str(i), "status": "queued", "large": i < 2} for i in range(10)]
    selected = batch.choose_next(jobs, [], 6)
    assert len(selected) == 6 and sum(j["large"] for j in selected) == 1
    for j in selected:
        j["status"] = "running"
    assert batch.choose_next(jobs, selected, 6) == []
    selected[0]["status"] = "finished"
    active = selected[1:]
    assert batch.choose_next(jobs, active, 6) == [jobs[1]]


def test_accounting_deduplicates_copies_and_keeps_unknowns(tmp_path):
    record = {
        "request_id": "one",
        "role": "worker",
        "status": "ok",
        "input_tokens": 5,
        "output_tokens": 3,
        "cost_usd": 0.01,
        "model_reported": "scripted",
    }
    for prefix in ["setup-001", "run-001"]:
        p = tmp_path / "studies/job/attempt-001" / prefix / ".mlca/requests/one.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"usage": record}))
    pending = {
        "request_id": "two",
        "role": "worker",
        "status": "pending",
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
    }
    (p.parent / "two.json").write_text(json.dumps({"usage": pending}))
    result = batch.accounting(batch.request_records(tmp_path))
    assert result["requests"] == 2 and result["known"]["cost_usd"] == 0.01
    assert result["partial"] and result["unknown"]["cost_usd"] == 1


def test_process_queue_and_resume_never_repeat_finished_studies(tmp_path, monkeypatch):
    code = tmp_path / "code"
    root = tmp_path / "experiments"
    (code / "rails").mkdir(parents=True)
    (code / "rails/VERSION").write_text("rails-0.3\n")
    (code / "benchmarks").mkdir()
    root.mkdir()
    entries = []
    for i in range(12):
        data = code / "benchmarks" / f"{i}.csv"
        data.write_text("feature\n1\n2\n")
        entries.append(
            {
                "id": f"table-{i}",
                "file": data.name,
                "role": "development",
                "rows": 600000 if i < 2 else 2,
                "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            }
        )
    monkeypatch.setattr(batch, "CODE", code)
    monkeypatch.setattr(batch, "ROOT", root)
    identity = {
        "code_commit": "fixture",
        "controller_commit": "fixture",
        "rails": {"version": "rails-0.3", "sha256": load(code / "rails").sha256},
    }
    monkeypatch.setattr(batch, "source_identity", lambda: identity)
    monkeypatch.setattr(batch, "entries", lambda: entries)
    monkeypatch.setattr(batch, "disk_needed", lambda job: 0)  # host disk is not under test
    monkeypatch.setattr(batch, "git_head", lambda _: "fixture")
    monkeypatch.setattr(batch, "runtime_identity", lambda _: {})
    original_run = subprocess.run
    monkeypatch.setattr(
        batch.subprocess,
        "run",
        lambda args, **kw: (
            type("Done", (), {"returncode": 0})()
            if args[0] == "git"
            else original_run(args, **kw)
        ),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "SYNTHETIC_NO_NETWORK")
    monkeypatch.setattr("mlca.sandbox.level.detect", lambda *_: "A")
    original_popen = subprocess.Popen
    children = []
    large = []
    peaks = []

    def launch(command, **kwargs):
        spec = json.loads(Path(command[-1]).read_text())
        assert "leakage_columns" not in spec and "targets" not in spec
        alive = [(p, l) for p, l in zip(children, large) if p.poll() is None]
        this_large = Path(spec["data"]).name in ["0.csv", "1.csv"]
        assert not this_large or not any(l for _, l in alive)
        peaks.append(len(alive) + 1)
        script = 'import json,sys,time;from pathlib import Path;s=json.load(open(sys.argv[1]));p=Path(s["attempt"]);p.mkdir(parents=True);time.sleep(.1);(p/"worker-result.json").write_text(json.dumps({"status":"failed","category":"synthetic_queue_test"}))'
        proc = original_popen([sys.executable, "-c", script, command[-1]], **kwargs)
        children.append(proc)
        large.append(this_large)
        return proc

    monkeypatch.setattr(batch.subprocess, "Popen", launch)
    original_sleep = time.sleep
    monkeypatch.setattr(batch.time, "sleep", lambda _: original_sleep(0.03))
    folder = root / "batch-01"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_batch",
            "--out",
            str(folder),
            "--batch",
            "1",
            "--model",
            "synthetic/no-network",
        ],
    )
    batch.main()
    state = json.loads((folder / "batch.json").read_text())
    assert len(children) == 12 and max(peaks) == 6
    assert all(j["status"] == "finished" for j in state["jobs"])
    assert state["accounting"]["requests"] == 0
    samples = [
        json.loads(line)
        for line in (folder / "resources.jsonl").read_text().splitlines()
    ]
    assert samples and samples[0]["disk_free_bytes"] > 0
    monkeypatch.setattr(sys, "argv", ["run_batch", "--out", str(folder), "--resume"])
    batch.main()
    assert len(children) == 12


def test_acceptance_and_ablation_require_prior_success(tmp_path):
    import pytest

    identity = {"code_commit": "v1", "rails": {"sha256": "hash"}}
    ids = {"a", "b"}
    target = tmp_path / "batch-02"
    with pytest.raises(ValueError, match="Acceptance"):
        batch.validate_stage(target, identity, "model", list(ids), ids, 3, False)
    first = tmp_path / "batch-01"
    first.mkdir()
    state = {
        **identity,
        "status": "finished",
        "model": "model",
        "rails_enabled": True,
        "replicates": 1,
        "jobs": [{"dataset": name, "result": {"success": True}} for name in ids],
    }
    (first / "batch.json").write_text(json.dumps(state))
    batch.validate_stage(target, identity, "model", list(ids), ids, 3, False)
    with pytest.raises(ValueError, match="Ablation"):
        batch.validate_stage(target, identity, "model", list(ids), ids, 1, True)
    state["replicates"] = 3
    state["jobs"] *= 3
    (first / "batch.json").write_text(json.dumps(state))
    batch.validate_stage(target, identity, "model", list(ids), ids, 1, True)
    with pytest.raises(ValueError, match="Ablation"):
        batch.validate_stage(
            target, {**identity, "code_commit": "v2"}, "model", list(ids), ids, 1, True
        )


def test_scored_studies_drop_never_published_stores(tmp_path):
    for project in ["setup-001", "run-001"]:
        for name in [".mlca/private", ".mlca/env", ".mlca/log", ".mlca/llm"]:
            (tmp_path / project / name).mkdir(parents=True)
            (tmp_path / project / name / "file").write_text("x")
    removed = batch.prune(tmp_path)
    assert removed == [
        "run-001/.mlca/private",
        "run-001/.mlca/env",
        "setup-001/.mlca/private",
        "setup-001/.mlca/env",
    ]
    assert (tmp_path / "run-001/.mlca/log/file").is_file()
    assert (tmp_path / "setup-001/.mlca/llm/file").is_file()


def test_large_tables_need_more_free_disk_before_starting():
    assert batch.disk_needed({"large": True}) > batch.disk_needed({"large": False}) >= 4 * 2**30

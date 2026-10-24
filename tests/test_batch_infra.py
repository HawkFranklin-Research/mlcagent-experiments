"""Infrastructure handling for cloud runs: environment preflight, machine loss, operator pause."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import env_preflight
import pytest
import run_batch as batch
from mlca.rails import load

ENV_FAIL = {
    "status": "failed",
    "category": "runtime_error:MlcaUserError",
    "furthest_event": "setup",
    "message": "Setup failed: MlcaUserError: Project environment command failed: Using Python 3.13.17 environment at: .mlca/env",
    "usage": {"requests": 2, "known": {"input_tokens": 900, "output_tokens": 200}},
}
CONNECTION_FAIL = {
    "status": "failed",
    "category": "provider_error",
    "furthest_event": "setup",
    "error_type": "MlcaUserError",
    "message": "Setup failed: MlcaUserError: Connection error.",
    "usage": {"known": {"input_tokens": 0, "output_tokens": 0}},
}
AGENT_FAIL = {
    "status": "failed",
    "category": "script_error",
    "furthest_event": "model_train",
    "message": "Repairs exhausted",
    "usage": {"known": {"input_tokens": 5000, "output_tokens": 900}},
}


def test_infrastructure_classifier():
    assert batch.infrastructure_reason({"result": ENV_FAIL}) == "environment_failure"
    assert (
        batch.infrastructure_reason({"result": CONNECTION_FAIL}) == "connection_failure"
    )
    assert batch.infrastructure_reason({"result": AGENT_FAIL}) is None
    mid_study_provider = {**AGENT_FAIL, "category": "provider_error"}
    assert batch.infrastructure_reason({"result": mid_study_provider}) is None
    assert batch.needs_operator({"result": mid_study_provider})
    assert batch.needs_operator({"result": ENV_FAIL})
    assert not batch.needs_operator({"result": AGENT_FAIL})


def harness(tmp_path, monkeypatch, jobs, worker_result):
    code = tmp_path / "code"
    root = tmp_path / "experiments"
    (code / "rails").mkdir(parents=True)
    (code / "rails/VERSION").write_text("rails-0.3\n")
    (code / "benchmarks").mkdir()
    root.mkdir()
    data = code / "benchmarks" / "t.csv"
    data.write_text("c\n1\n")
    entries = [
        {
            "id": "t",
            "file": data.name,
            "role": "development",
            "rows": 1,
            "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        }
    ]
    monkeypatch.setattr(batch, "CODE", code)
    monkeypatch.setattr(batch, "ROOT", root)
    identity = {
        "code_commit": "c",
        "controller_commit": "ctl",
        "controller_tree": "tree",
        "rails": {"version": "rails-0.3", "sha256": load(code / "rails").sha256},
    }
    monkeypatch.setattr(batch, "source_identity", lambda: identity)
    monkeypatch.setattr(batch, "entries", lambda *a, **k: entries)
    monkeypatch.setattr(batch, "disk_needed", lambda _: 0)
    monkeypatch.setattr(batch, "runtime_identity", lambda _: {})
    monkeypatch.setattr(batch, "frozen", lambda _: [])
    monkeypatch.setattr(batch, "probe_endpoint", lambda *_: True)
    monkeypatch.setenv("OPENROUTER_API_KEY", "SYNTHETIC_KEY")
    monkeypatch.setattr("mlca.sandbox.level.detect", lambda *_: "A")
    folder = root / "batch-infra"
    (folder / ".control").mkdir(parents=True)
    (root / "batch-start.json").write_text(json.dumps({"started_at": batch.utc()}))
    state = {
        **identity,
        "batch": 1,
        "stage": "development",
        "role": "development",
        "manifest": None,
        "data_root": None,
        "provider": "openrouter",
        "endpoint": "https://openrouter.ai/api/v1",
        "comparison_plan_sha256": None,
        "started_at": batch.utc(),
        "model": "synthetic/model",
        "reasoning_effort": None,
        "concurrency": 1,
        "deadline_s": 5400,
        "rpm": 60,
        "rails_enabled": True,
        "replicates": 1,
        "threads": 1,
        "memory_limit_mib": None,
        "memory_limit": "not_enforced",
        "runtime": {},
        "sandbox": "A",
        "model_call_deadline_s": 300,
        "temperature": 0,
        "max_output_tokens": 16384,
        "structured_outputs": True,
        "sdk_timeout_s": 90,
        "rate_limit_dir": str(folder / ".control/rate"),
        "machine": {"cpus": 2},
        "jobs": jobs,
    }
    (folder / "batch.json").write_text(json.dumps(state))
    original_popen = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        spec = json.loads(Path(command[-1]).read_text())
        Path(spec["attempt"]).mkdir(parents=True, exist_ok=True)
        script = (
            "import json,sys;from pathlib import Path;s=json.load(open(sys.argv[1]));"
            f"(Path(s['attempt'])/'worker-result.json').write_text(json.dumps({worker_result!r}))"
        )
        proc = original_popen([sys.executable, "-c", script, command[-1]], **kwargs)
        children.append(proc)
        return proc

    monkeypatch.setattr(batch.subprocess, "Popen", launch)
    return folder, children


def job(number, status, **extra):
    return {
        "id": f"t-{number:03d}",
        "dataset": "t",
        "replicate": number,
        "large": False,
        "status": status,
        "first_started": 1000.0,
        "attempts": [f"studies/t-{number:03d}/attempt-001"],
        **extra,
    }


def queued(number):
    return {
        "id": f"t-{number:03d}",
        "dataset": "t",
        "replicate": number,
        "large": False,
        "status": "queued",
        "attempts": [],
    }


def test_resume_reruns_infrastructure_losses_only(tmp_path, monkeypatch):
    jobs = [
        job(1, "finished", result=ENV_FAIL),
        job(2, "interrupted", stop_reason="machine_shutdown"),
        job(3, "running", pid=999999999, process_start="0"),
        job(4, "finished", result=AGENT_FAIL),
    ]
    folder, children = harness(
        tmp_path,
        monkeypatch,
        jobs,
        {"status": "success", "category": "clean", "workflow_completed": True},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_batch",
            "--out",
            str(folder),
            "--resume",
            "--retry-infrastructure-failures",
        ],
    )
    batch.main()
    state = json.loads((folder / "batch.json").read_text())
    by_id = {j["id"]: j for j in state["jobs"]}
    assert [r["reason"] for r in by_id["t-001"]["retries"]] == ["environment_failure"]
    assert [r["reason"] for r in by_id["t-002"]["retries"]] == ["machine_shutdown"]
    assert [r["reason"] for r in by_id["t-003"]["retries"]] == ["machine_lost"]
    for number in (1, 2, 3):
        rerun = by_id[f"t-{number:03d}"]
        assert rerun["status"] == "finished"
        assert rerun["attempts"][-1].endswith("attempt-002")
        assert rerun["first_started_original"] == 1000.0
        assert (
            rerun["first_started"] > 1000.0
        )  # deadline clock restarted for the new attempt
    assert "retries" not in by_id["t-004"]
    assert by_id["t-004"]["result"]["category"] == "script_error"
    assert len(children) == 3
    assert state["env_preflights"][-1]["python"] == "test"
    report = (folder / "REPORT.md").read_text()
    assert "## Retried attempts" in report and "## Environment preflight" in report


def test_second_retry_for_same_reason_is_refused(tmp_path, monkeypatch):
    prior = {
        "retry_of": "studies/t-001/attempt-001",
        "reason": "environment_failure",
        "at": "x",
    }
    folder, _ = harness(
        tmp_path,
        monkeypatch,
        [job(1, "finished", result=ENV_FAIL, retries=[prior])],
        {"status": "success"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_batch",
            "--out",
            str(folder),
            "--resume",
            "--retry-infrastructure-failures",
        ],
    )
    with pytest.raises(
        ValueError, match="already retried once for environment_failure"
    ):
        batch.main()


def test_provider_error_pauses_scheduler(tmp_path, monkeypatch):
    folder, children = harness(
        tmp_path,
        monkeypatch,
        [queued(1), queued(2)],
        {
            **AGENT_FAIL,
            "category": "provider_error",
            "message": "Invalid request: maximum context length",
        },
    )
    monkeypatch.setattr(sys, "argv", ["run_batch", "--out", str(folder), "--resume"])
    batch.main()
    state = json.loads((folder / "batch.json").read_text())
    assert state["status"] == "paused"
    assert len(children) == 1
    assert state["operator_pauses"][0]["reason"] == "provider_error"
    assert [j["status"] for j in state["jobs"]] == ["finished", "queued"]
    assert "Scheduler paused for the operator" in (folder / "REPORT.md").read_text()


def test_env_preflight_package_sets_cover_all_templates():
    sets = env_preflight.package_sets()
    assert set(sets) == {"default", "lifelines"}
    assert "lifelines" in sets["lifelines"] and "lifelines" not in sets["default"]


def test_env_preflight_rejects_version_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(env_preflight, "REFERENCE", tmp_path)
    monkeypatch.setattr(env_preflight, "ROOT", tmp_path.parent)
    monkeypatch.setattr(env_preflight, "package_sets", lambda: {"default": ["numpy"]})
    (tmp_path / "default.requirements.txt").write_text("numpy==2.5.4\n")
    monkeypatch.setattr(env_preflight, "build", lambda _: ["numpy==2.5.4"])
    assert env_preflight.preflight()["sets"]["default"]["matches_reference"]
    monkeypatch.setattr(env_preflight, "build", lambda _: ["numpy==2.6.0"])
    with pytest.raises(ValueError, match="differs from"):
        env_preflight.preflight()

    def broken(_):
        raise RuntimeError("only cp312 wheels")

    monkeypatch.setattr(env_preflight, "build", broken)
    with pytest.raises(ValueError, match="Environment preflight failed for 'default'"):
        env_preflight.preflight()

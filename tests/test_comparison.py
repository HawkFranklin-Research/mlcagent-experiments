import json
import sys
from pathlib import Path

import pytest
import run_batch as batch
from comparison import (
    build_plan,
    digest,
    read_plan,
    select_jobs,
    validate_plan,
    validate_state,
)
from mlca.paths import atomic_json
from pool_batches import pool, write_summary
from study_resources import enter_group, memory_events, prepare_group


@pytest.fixture
def plan():
    return build_plan(
        {
            "code_commit": "code",
            "controller_commit": "controller",
            "rails": {"version": "rails-0.9", "sha256": "rails"},
        },
        [{"id": "a", "sha256": "hash-a"}, {"id": "b", "sha256": "hash-b"}],
        provider="deepseek",
        model="deepseek-flash",
        effort="low",
        runtime={"lock": "fixture"},
    )


def state_for(plan, stage, jobs=None, success=True):
    from comparison import BATCH_SETTINGS

    return {
        **plan["identity"],
        **{key: plan[pkey] for key, pkey in BATCH_SETTINGS.items()},
        "comparison_plan_sha256": plan["plan_sha256"],
        "stage": stage,
        "rails_enabled": stage == "acceptance",
        "replicates": 3 if stage == "acceptance" else 1,
        "status": "finished",
        "jobs": [
            {
                **j,
                "status": "finished",
                "result": {
                    "success": success,
                    "usage": {"requests": 0},
                    "split_parameters": {"seed": 42},
                },
            }
            for j in (jobs or select_jobs(plan, stage))
        ],
    }


def store(tmp_path, name, state):
    folder = tmp_path / name
    atomic_json(folder / "batch.json", state)
    return folder


def test_plan_jobs_pairing_and_shards(plan, tmp_path):
    assert len(plan["jobs"]) == 8
    assert len({j["study_id"] for j in plan["jobs"]}) == 8
    assert all(j["protocol_seed"] == 42 for j in plan["jobs"])
    assert plan["spending_ceiling_usd"] is None
    for stage in ["acceptance", "ablation"]:
        parts = [select_jobs(plan, stage, shard=f"{i}/2") for i in [1, 2]]
        ids = [[j["study_id"] for j in p] for p in parts]
        assert not set(ids[0]) & set(ids[1])
        assert set(ids[0] + ids[1]) == {j["study_id"] for j in select_jobs(plan, stage)}
    p = tmp_path / "plan.json"
    atomic_json(p, plan)
    assert read_plan(p) == plan
    altered = {**plan, "model": "other"}
    atomic_json(p, altered)
    with pytest.raises(ValueError, match="hash"):
        read_plan(p)


@pytest.mark.parametrize("shard", ["0/2", "3/2", "a/b", "1/0", "1"])
def test_invalid_shard(plan, shard):
    with pytest.raises(ValueError, match="Shard"):
        select_jobs(plan, "acceptance", shard=shard)


def test_no_unknown_or_duplicate_tables(plan):
    for tables in [["unknown"], ["a", "a"]]:
        with pytest.raises(ValueError):
            select_jobs(plan, "acceptance", tables=tables)


def test_validate_plan_rejects_source_runtime_and_data_changes(plan):
    datasets = plan["datasets"]
    validate_plan(plan, plan["identity"], datasets, plan["runtime"])
    with pytest.raises(ValueError, match="identical"):
        validate_plan(
            plan, {**plan["identity"], "code_commit": "new"}, datasets, plan["runtime"]
        )
    with pytest.raises(ValueError, match="identical"):
        validate_plan(plan, plan["identity"], datasets, {})
    with pytest.raises(ValueError, match="manifest"):
        validate_plan(
            plan, plan["identity"], [{"id": "a", "sha256": "new"}], plan["runtime"]
        )


def test_pool_keeps_failures_missing_and_candidate_outcome(plan, tmp_path):
    acceptance = state_for(plan, "acceptance", success=False)
    ablation = state_for(plan, "ablation")
    a = store(tmp_path, "acceptance", acceptance)
    b = store(tmp_path, "ablation", ablation)
    result = pool(plan, [a, b])
    assert result["candidate_acceptance"] == "not_accepted"
    assert result["arms"]["acceptance"]["failed"] == 6
    assert result["arms"]["ablation"]["qualified"] == 2
    assert not result["missing_studies"]
    assert all(p["split_parameters_match"] for p in result["pairs"])
    assert result["accounting_complete"]
    result = pool(plan, [b])
    assert result["candidate_acceptance"] == "pending"
    assert len(result["missing_studies"]) == 6
    assert result["arms"]["ablation"]["finished"] == 2
    write_summary(tmp_path / "summary", result)
    assert (tmp_path / "summary/REPORT.md").exists()


def test_pool_duplicates_rejected(plan, tmp_path):
    folder = store(tmp_path, "a", state_for(plan, "acceptance"))
    with pytest.raises(ValueError, match="Duplicate study"):
        pool(plan, [folder, folder])


@pytest.mark.parametrize(
    "key,value",
    [
        ("code_commit", "other"),
        ("rails", {"sha256": "other"}),
        ("model", "other"),
        ("provider", "openrouter"),
        ("endpoint", "https://other.example"),
        ("reasoning_effort", "high"),
        ("threads", 4),
        ("memory_limit_mib", 2048),
        ("rails_enabled", False),
        ("runtime", {}),
        ("deadline_s", 1),
    ],
)
def test_pool_refuses_incompatible_conditions(plan, tmp_path, key, value):
    state = state_for(plan, "acceptance")
    state[key] = value
    folder = store(tmp_path, "incompatible", state)
    with pytest.raises(ValueError):
        pool(plan, [folder])


def test_pool_missing_usage_unknown_not_zero(plan, tmp_path):
    state = state_for(plan, "ablation")
    state["jobs"][0]["result"].pop("usage")
    folder = store(tmp_path, "missing", state)
    result = pool(plan, [folder])
    assert result["accounting"] is None
    assert not result["accounting_complete"]


def test_pool_different_actual_splits_are_visible(plan, tmp_path):
    a, b = state_for(plan, "acceptance"), state_for(plan, "ablation")
    b["jobs"][0]["result"]["split_parameters"]["seed"] = 99
    result = pool(plan, [store(tmp_path, "a", a), store(tmp_path, "b", b)])
    assert result["pairs"][0]["split_parameters_match"] is False


def test_enforcement_mismatch_refused(plan, tmp_path):
    state = state_for(plan, "ablation")
    state["jobs"][0]["result"]["enforced_resources"] = {"threads": 9}
    with pytest.raises(ValueError, match="enforcement"):
        pool(plan, [store(tmp_path, "a", state)])


def test_resource_memory_limit_fails_closed(tmp_path, monkeypatch):
    # File-system stub tests the cgroup protocol, not actual kernel containment.
    monkeypatch.setattr("study_resources.require_cgroup_v2", lambda _: None)
    with pytest.raises(OSError):
        prepare_group(tmp_path / "missing", "study", 4096)
    (tmp_path / "cgroup.subtree_control").write_text("cpu")
    with pytest.raises(ValueError, match="delegated"):
        prepare_group(tmp_path, "study", 4096)
    (tmp_path / "cgroup.subtree_control").write_text("memory cpu")
    group = prepare_group(tmp_path, "study", 4096)
    assert (group / "memory.max").read_text() == str(4096 * 1024 * 1024)
    assert (group / "memory.swap.max").read_text() == "0"
    enter_group(group, 4096)
    assert int((group / "cgroup.procs").read_text()) > 0
    with pytest.raises(ValueError, match="differs"):
        enter_group(group, 2048)
    (group / "memory.events").write_text("oom 1\noom_kill 1\n")
    assert memory_events(group)["oom_kill"] == 1


@pytest.mark.parametrize("stage", ["acceptance", "ablation"])
def test_planned_controller_routes_provider_and_keeps_each_arm_fixed(
    plan, tmp_path, monkeypatch, stage
):
    # Process stubs only: synthetic outcomes, no analyses, network or MLCA execution.
    import subprocess
    import time

    code = tmp_path / "code"
    (code / "benchmarks").mkdir(parents=True)
    datasets = []
    for name in ["a", "b"]:
        path = code / "benchmarks" / (name + ".csv")
        path.write_text("x\n1\n")
        datasets.append(
            {
                "id": name,
                "file": path.name,
                "rows": 1,
                "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            }
        )
    plan = build_plan(
        plan["identity"],
        datasets,
        provider="deepseek",
        model="deepseek-flash",
        effort="low",
        runtime=plan["runtime"],
    )
    plan_path = tmp_path / "plan.json"
    atomic_json(plan_path, plan)
    monkeypatch.setattr(batch, "CODE", code)
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "source_identity", lambda: plan["identity"])
    monkeypatch.setattr(batch, "runtime_identity", lambda _: plan["runtime"])
    monkeypatch.setattr(batch, "entries", lambda *a, **k: datasets)
    monkeypatch.setattr("mlca.sandbox.level.detect", lambda *_: "A")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "SYNTHETIC_NO_NETWORK")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    popen = subprocess.Popen
    launches = []

    def launch(command, **kw):
        spec = json.loads(Path(command[-1]).read_text())
        launches.append(spec)
        assert spec["config"]["providers"]["development"]["kind"] == "deepseek"
        assert (
            spec["config"]["providers"]["development"]["api_key_env"]
            == "DEEPSEEK_API_KEY"
        )
        assert spec["config"]["rails_enabled"] is (stage == "acceptance")
        assert spec["config"]["budgets"]["session_max_usd"] is None
        assert len(spec["affinity"]) == 2
        rate_dir = spec["config"]["providers"]["development"]["rate_limit_dir"]
        assert plan["plan_sha256"] in rate_dir
        assert "mlca-comparison-rate" in rate_dir
        script = 'import json,sys;from pathlib import Path;s=json.load(open(sys.argv[1]));p=Path(s["attempt"]);p.mkdir(parents=True);(p/"worker-result.json").write_text(json.dumps({"status":"failed","category":"synthetic"}))'
        return popen([sys.executable, "-c", script, command[-1]], **kw)

    monkeypatch.setattr(batch.subprocess, "Popen", launch)
    sleep = time.sleep
    monkeypatch.setattr(batch.time, "sleep", lambda _: sleep(0.03))
    argv = [
        "run_batch",
        "--out",
        str(tmp_path / stage),
        "--comparison-plan",
        str(plan_path),
        "--stage",
        stage,
    ]
    if stage == "ablation":
        argv += ["--provisional-ablation", "--shard", "1/2"]
    monkeypatch.setattr(sys, "argv", argv)
    batch.main()
    state = json.loads((tmp_path / stage / "batch.json").read_text())
    assert len(launches) == (6 if stage == "acceptance" else 1)
    validate_state(plan, state)
    assert state["status"] == "finished"
    assert state["memory_limit"] == "not_enforced"
    # Resume uses stored credentials/arm without requiring repeated flags.
    monkeypatch.setattr(
        sys, "argv", ["run_batch", "--out", str(tmp_path / stage), "--resume"]
    )
    batch.main()
    assert len(launches) == (6 if stage == "acceptance" else 1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_batch",
            "--out",
            str(tmp_path / stage),
            "--resume",
            "--provider",
            "openrouter",
        ],
    )
    with pytest.raises(ValueError, match="Resume cannot change"):
        batch.main()


def test_worker_enforcement_failure_prevents_any_agent_run(tmp_path, monkeypatch):
    monkeypatch.setattr("study_resources.require_cgroup_v2", lambda _: None)
    from mlca.autonomous import runner

    def forbidden(*args, **kwargs):
        raise AssertionError("Agent must not start when memory enforcement fails")

    monkeypatch.setattr(runner, "run", forbidden)
    spec = tmp_path / "worker.json"
    atomic_json(
        spec,
        {
            "attempt": str(tmp_path / "study/attempt-001"),
            "environment": {"OMP_NUM_THREADS": "2"},
            "affinity": [],
            "cgroup": str(tmp_path / "missing-group"),
            "memory_limit_mib": 2048,
        },
    )
    batch.worker(spec)
    result = json.loads((tmp_path / "study/attempt-001/worker-result.json").read_text())
    assert result["status"] == "failed"
    assert result["error_type"] == "FileNotFoundError"
    assert not (tmp_path / "study/attempt-001-enforced-resources.json").exists()


def test_rehashed_but_inconsistent_plan_refused(plan, tmp_path):
    plan["jobs"][0]["rails_enabled"] = False
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    plan["plan_sha256"] = digest(body)
    path = tmp_path / "bad-plan.json"
    atomic_json(path, plan)
    with pytest.raises(ValueError, match="inconsistent"):
        read_plan(path)


def test_plan_creation_offline_with_development_evidence(plan, tmp_path, monkeypatch):
    import plan_comparison

    development = tmp_path / "development"
    state = {
        "status": "finished",
        "rails_enabled": True,
        "model": "prior-router-model",
        "jobs": [{"dataset": name, "result": {"success": True}} for name in ["a", "b"]],
    }
    atomic_json(development / "batch-01/batch.json", state)
    monkeypatch.setattr(batch, "source_identity", lambda: plan["identity"])
    monkeypatch.setattr(batch, "entries", lambda *a, **k: plan["datasets"])
    monkeypatch.setattr(plan_comparison, "runtime_identity", lambda _: plan["runtime"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan_comparison",
            "--out",
            str(tmp_path / "plan.json"),
            "--development-root",
            str(development),
            "--provider",
            "deepseek",
            "--model",
            "deepseek-flash",
        ],
    )
    plan_comparison.main()
    created = read_plan(tmp_path / "plan.json")
    assert created["provider"] == "deepseek"
    assert created["development_evidence"][0]["provider"] == "openrouter"
    assert created["spending_ceiling_usd"] is None
    assert len(created["jobs"]) == 8


def test_normal_folder_cannot_claim_kernel_memory_enforcement(tmp_path):
    (tmp_path / "cgroup.subtree_control").write_text("memory")
    with pytest.raises(ValueError, match="real cgroup-v2"):
        prepare_group(tmp_path, "fake-group", 2048)

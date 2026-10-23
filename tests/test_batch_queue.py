import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import run_batch as batch
from comparison import provider_kind
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


@pytest.mark.parametrize(
    "provider,structured,freeze,replicates,subset",
    [
        ("openrouter", "on", False, 1, False),
        ("ollama", "off", False, 1, False),
        ("ollama", "off", True, 3, False),
        ("ollama", "off", True, 1, True),
    ],
)
def test_process_queue_and_resume_never_repeat_finished_studies(
    tmp_path, monkeypatch, provider, structured, freeze, replicates, subset
):
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
    monkeypatch.setattr(batch, "entries", lambda *a, **k: entries)
    monkeypatch.setattr(
        batch, "disk_needed", lambda job: 0
    )  # host disk is not under test
    monkeypatch.setattr(batch, "git_head", lambda _: "fixture")
    monkeypatch.setattr(batch, "runtime_identity", lambda _: {})
    monkeypatch.setattr(batch, "frozen", lambda _: ["freeze-test"] if freeze else [])
    if freeze:
        monkeypatch.setattr(
            batch,
            "validate_stage",
            lambda *a, **k: pytest.fail(
                "Replication must skip development/acceptance gates"
            ),
        )
        (root / "batch-start.json").write_text(
            json.dumps({"started_at": "2000-01-01T00:00:00+00:00"})
        )
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
    monkeypatch.setenv(
        "OPENROUTER_API_KEY" if provider == "openrouter" else "OLLAMA_API_KEY",
        "SYNTHETIC_NO_NETWORK",
    )
    monkeypatch.setattr("mlca.sandbox.level.detect", lambda *_: "A")
    original_popen = subprocess.Popen
    children = []
    large = []
    peaks = []

    def launch(command, **kwargs):
        spec = json.loads(Path(command[-1]).read_text())
        assert "leakage_columns" not in spec and "targets" not in spec
        config = spec["config"]["providers"]["development"]
        assert config["kind"] == provider_kind(provider)
        assert config["base_url"] == batch.PROVIDERS[provider][0]
        assert config["api_key_env"] == batch.PROVIDERS[provider][1]
        assert config["structured_outputs"] is (structured == "on")
        assert config["timeout_s"] == (300 if provider == "ollama" else 90)
        assert config["call_deadline_s"] == (600 if provider == "ollama" else 300)
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
            "minimax-m3.1-flash" if provider == "ollama" else "synthetic/no-network",
            "--provider",
            provider,
            "--structured-outputs",
            structured,
            "--replicates",
            str(replicates),
        ]
        + (["--tables", "table-0"] if subset else []),
    )
    batch.main()
    state = json.loads((folder / "batch.json").read_text())
    expected = (1 if subset else 12) * replicates
    assert len(children) == expected and max(peaks) == (
        1 if subset or provider == "ollama" else 6
    )
    assert state["stage"] == ("replication" if freeze else "development")
    assert state["structured_outputs"] is (structured == "on")
    assert state["sdk_timeout_s"] == (300 if provider == "ollama" else 90)
    assert state["model_call_deadline_s"] == (600 if provider == "ollama" else 300)
    assert all(j["status"] == "finished" for j in state["jobs"])
    assert state["accounting"]["requests"] == 0
    samples = [
        json.loads(line)
        for line in (folder / "resources.jsonl").read_text().splitlines()
    ]
    assert samples and samples[0]["disk_free_bytes"] > 0
    monkeypatch.setattr(sys, "argv", ["run_batch", "--out", str(folder), "--resume"])
    batch.main()
    assert len(children) == expected
    for option in [
        ["--structured-outputs", "on" if structured == "off" else "off"],
        ["--structured-outputs=" + ("on" if structured == "off" else "off")],
    ]:
        monkeypatch.setattr(
            sys, "argv", ["run_batch", "--out", str(folder), "--resume"] + option
        )
        with pytest.raises(
            ValueError, match="Resume cannot change recorded structured_outputs"
        ):
            batch.main()
    for flag, key in [
        ("--sdk-timeout-s", "sdk_timeout_s"),
        ("--model-call-deadline-s", "model_call_deadline_s"),
    ]:
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_batch", "--out", str(folder), "--resume", flag, str(state[key] + 1)],
        )
        with pytest.raises(ValueError, match="Resume cannot change recorded " + key):
            batch.main()
    assert len(children) == expected


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
    assert (
        batch.disk_needed({"large": True})
        > batch.disk_needed({"large": False})
        >= 4 * 2**30
    )


def test_preserve_split_manifest_before_pruning(tmp_path):
    import pytest
    from mlca.log.store import Store

    project = tmp_path / "run-001"
    source = project / ".mlca/data/splits/one/split_manifest.csv"
    source.parent.mkdir(parents=True)
    source.write_text("row_id,partition\n0,train\n1,val\n2,test\n")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    store = Store(project)
    store.append(
        "ProjectCreated",
        {
            "project_id": "synthetic",
            "mlca_version": "test",
            "template_id": "test",
            "template_version": "1",
            "template_sha256": "x",
        },
    )
    store.append(
        "SplitFrozen",
        {
            "split_id": "one",
            "sha256": sha,
            "manifest": {"path": str(source.relative_to(project)), "sha256": sha},
            "train": {},
            "val": {},
            "vault": {},
            "parameters": {},
            "train_index_sha256": "x",
            "counts": {"train": 1, "val": 1, "test": 1},
        },
    )
    retained = batch.preserve_split_manifests(tmp_path)
    assert retained[0]["sha256"] == sha
    assert (tmp_path / "split_manifest-run-001.csv").read_bytes() == source.read_bytes()
    batch.prune(tmp_path)
    assert (tmp_path / "split_manifest-run-001.csv").is_file()
    source.write_text("row_id,partition\n0,train\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        batch.preserve_split_manifests(tmp_path)


def test_controller_provider_kind_mapping():
    assert provider_kind("ollama") == "openai_compatible"
    assert provider_kind("deepseek") == "deepseek"
    assert provider_kind("openrouter") == "openrouter"


def test_untagged_full_replicates_keep_acceptance_gate(tmp_path, monkeypatch):
    code, root = tmp_path / "code", tmp_path / "experiments"
    root.mkdir()
    monkeypatch.setattr(batch, "CODE", code)
    monkeypatch.setattr(batch, "ROOT", root)
    monkeypatch.setattr(
        batch,
        "source_identity",
        lambda: {"code_commit": "fixture", "rails": {"sha256": "fixture"}},
    )
    monkeypatch.setattr(
        batch, "entries", lambda *a, **k: [{"id": f"table-{i}"} for i in range(12)]
    )
    monkeypatch.setattr(batch, "frozen", lambda _: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_batch",
            "--out",
            str(root / "batch-02"),
            "--batch",
            "2",
            "--model",
            "synthetic/no-network",
            "--replicates",
            "3",
        ],
    )
    with pytest.raises(ValueError, match="Acceptance"):
        batch.main()


@pytest.fixture
def identity_repositories(tmp_path, monkeypatch):
    """Real temporary Git repositories; no study, data or inference execution."""
    code, root = tmp_path / "code", tmp_path / "experiments"
    for repo in [code, root]:
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (code / "rails").mkdir()
    (code / "rails/VERSION").write_text("rails-0.3\n")
    (code / "src").mkdir()
    (code / "src/engine.py").write_text("# synthetic tracked source\n")
    (root / "scripts").mkdir()
    (root / "scripts/run_batch.py").write_text("# synthetic tracked controller\n")
    # Exercise the actual repository ignore rules rather than a test-only bypass.
    (root / ".gitignore").write_bytes(
        (Path(__file__).parents[1] / ".gitignore").read_bytes()
    )
    for repo in [code, root]:
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=MLCA Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "Synthetic source identity fixture",
            ],
            check=True,
        )
    monkeypatch.setattr(batch, "CODE", code)
    monkeypatch.setattr(batch, "ROOT", root)
    return code, root


def test_local_keepalive_does_not_change_runtime_identity(identity_repositories):
    _code, root = identity_repositories
    expected = batch.source_identity()
    helper = root / "scripts/codespace_keepalive.sh"
    helper.write_text("# synthetic local monitor, never executed\n")
    assert batch.source_identity() == expected
    helper.write_text("# changed local monitor, still never executed\n")
    assert batch.source_identity() == expected


@pytest.mark.parametrize(
    "repository,relative",
    [
        ("experiments", "scripts/untracked_runtime.py"),
        ("experiments", "scripts/other_helper.sh"),
        ("code", "src/untracked_engine.py"),
        ("code", "templates/untracked.yaml"),
        ("code", "rails/untracked.yaml"),
    ],
)
def test_untracked_runtime_still_blocks(identity_repositories, repository, relative):
    code, root = identity_repositories
    path = (root if repository == "experiments" else code) / relative
    path.parent.mkdir(exist_ok=True)
    path.write_text("# synthetic untracked runtime\n")
    with pytest.raises(ValueError, match="Untracked runtime source"):
        batch.source_identity()


@pytest.mark.parametrize(
    "repository,relative",
    [
        ("experiments", "scripts/run_batch.py"),
        ("code", "src/engine.py"),
    ],
)
def test_tracked_changes_still_block(identity_repositories, repository, relative):
    code, root = identity_repositories
    path = (root if repository == "experiments" else code) / relative
    path.write_text("# changed synthetic tracked source\n")
    with pytest.raises(ValueError, match="Tracked working tree must be clean"):
        batch.source_identity()


@pytest.mark.parametrize("flag", ["--sdk-timeout-s", "--model-call-deadline-s"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_call_timeouts_must_be_positive_and_finite(tmp_path, monkeypatch, flag, value):
    monkeypatch.setattr(
        batch,
        "source_identity",
        lambda: pytest.fail("Invalid timeout must fail before preflight"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_batch", "--out", str(tmp_path / "unused"), flag + "=" + value],
    )
    with pytest.raises(ValueError, match="positive and finite"):
        batch.main()


def test_provider_failure_summary_is_payload_free():
    failure = {
        "request_id": "one",
        "status": "error",
        "error_type": "APITimeoutError",
        "category": "provider_error",
        "status_code": None,
        "latency_ms": 90001,
        "pacing_wait_s": 1,
        "rate_limit_waits": 0,
        "request": "DO_NOT_COPY",
        "headers": "DO_NOT_COPY",
    }
    errors = batch.provider_errors([failure, {"request_id": "two", "status": "ok"}])
    assert len(errors) == 1 and errors[0]["latency_ms"] == 90001
    assert errors[0]["status_code"] is None
    assert "DO_NOT_COPY" not in json.dumps(errors)


def test_local_ollama_launcher_dry_run():
    import shlex

    script = Path(__file__).parents[1] / "scripts/run_ollama_local.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    completed = subprocess.run(
        ["bash", str(script), "--dry-run"], check=True, capture_output=True, text=True
    )
    command = shlex.split(completed.stdout)

    def setting(flag):
        return command[command.index(flag) + 1]

    assert setting("--provider") == "ollama"
    assert setting("--model") == "minimax-m3.1-flash"
    assert setting("--sdk-timeout-s") == "300"
    assert setting("--model-call-deadline-s") == "600"
    assert setting("--concurrency") == "1"
    assert setting("--tables") == "breast_cancer_diagnostic"
    assert Path(setting("--out")).is_relative_to(
        Path.home() / ".cache/mlca-smoke-ollama"
    )
    assert "--resume" not in command


def test_pause_drains_then_resume_continues(tmp_path, monkeypatch):
    code = tmp_path / "code"
    root = tmp_path / "experiments"
    (code / "rails").mkdir(parents=True)
    (code / "rails/VERSION").write_text("rails-0.3\n")
    (code / "benchmarks").mkdir()
    root.mkdir()
    entries = []
    for i in range(2):
        data = code / "benchmarks" / f"tab_{i}.csv"
        data.write_text("f1\n1\n2\n")
        entries.append({
            "id": f"tab_{i}",
            "file": data.name,
            "role": "development",
            "rows": 10,
            "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        })
    monkeypatch.setattr(batch, "CODE", code)
    monkeypatch.setattr(batch, "ROOT", root)
    identity = {
        "code_commit": "fix_code",
        "controller_commit": "fix_ctl",
        "controller_tree": "fix_tree",
        "rails": {"version": "rails-0.3", "sha256": load(code / "rails").sha256},
    }
    monkeypatch.setattr(batch, "source_identity", lambda: identity)
    monkeypatch.setattr(batch, "entries", lambda *a, **k: entries)
    monkeypatch.setattr(batch, "disk_needed", lambda _: 0)
    monkeypatch.setattr(batch, "git_head", lambda _: "fix_ctl")
    monkeypatch.setattr(batch, "git_tree", lambda *_: "fix_tree")
    monkeypatch.setattr(batch, "runtime_identity", lambda _: {})
    monkeypatch.setattr(batch, "frozen", lambda _: [])
    monkeypatch.setenv("OPENROUTER_API_KEY", "SYNTHETIC_KEY")
    monkeypatch.setattr("mlca.sandbox.level.detect", lambda *_: "A")

    folder = root / "batch-pause-test"
    original_popen = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        spec = json.loads(Path(command[-1]).read_text())
        p = Path(spec["attempt"])
        p.mkdir(parents=True, exist_ok=True)
        if len(children) == 0:
            (folder / ".control/PAUSE").touch()
        script = 'import json,sys,time;from pathlib import Path;s=json.load(open(sys.argv[1]));p=Path(s["attempt"]);time.sleep(0.05);(p/"worker-result.json").write_text(json.dumps({"status":"failed","category":"test_done"}))'
        proc = original_popen([sys.executable, "-c", script, command[-1]], **kwargs)
        children.append(proc)
        return proc

    monkeypatch.setattr(batch.subprocess, "Popen", launch)
    orig_sleep = time.sleep
    monkeypatch.setattr(batch.time, "sleep", lambda _: orig_sleep(0.01))

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
            "synthetic/model",
            "--concurrency",
            "1",
            "--tables",
            "tab_0",
            "tab_1",
        ],
    )
    batch.main()
    state = json.loads((folder / "batch.json").read_text())
    assert state["status"] == "paused"
    assert len(state.get("pauses", [])) == 1
    assert "paused_at" in state["pauses"][0]
    assert len(children) == 1
    assert state["jobs"][0]["status"] == "finished"
    assert state["jobs"][1]["status"] == "queued"

    monkeypatch.setattr(sys, "argv", ["run_batch", "--out", str(folder), "--resume"])
    batch.main()
    state2 = json.loads((folder / "batch.json").read_text())
    assert state2["status"] == "finished"
    assert not (folder / ".control/PAUSE").exists()
    assert state2["pauses"][0].get("resumed_at") is not None
    assert len(children) == 2
    assert all(j["status"] == "finished" for j in state2["jobs"])


def test_controller_upgrade_accepted_and_refused(tmp_path, monkeypatch):
    code = tmp_path / "code"
    root = tmp_path / "experiments"
    (code / "rails").mkdir(parents=True)
    (code / "rails/VERSION").write_text("rails-0.3\n")
    (code / "benchmarks").mkdir()
    root.mkdir()
    data = code / "benchmarks" / "t.csv"
    data.write_text("c\n1\n")
    entries = [{
        "id": "t",
        "file": data.name,
        "role": "development",
        "rows": 1,
        "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    }]
    monkeypatch.setattr(batch, "CODE", code)
    monkeypatch.setattr(batch, "ROOT", root)
    rails_info = {"version": "rails-0.3", "sha256": load(code / "rails").sha256}
    initial_identity = {
        "code_commit": "code_A",
        "controller_commit": "ctl_A",
        "controller_tree": "tree_A",
        "rails": rails_info,
    }
    monkeypatch.setattr(batch, "source_identity", lambda: initial_identity)
    monkeypatch.setattr(batch, "entries", lambda *a, **k: entries)
    monkeypatch.setattr(batch, "disk_needed", lambda _: 0)
    monkeypatch.setattr(batch, "git_head", lambda _: "ctl_A")
    monkeypatch.setattr(batch, "git_tree", lambda *_: "tree_A")
    monkeypatch.setattr(batch, "runtime_identity", lambda _: {})
    monkeypatch.setattr(batch, "frozen", lambda _: [])
    monkeypatch.setenv("OPENROUTER_API_KEY", "SYNTHETIC_KEY")
    monkeypatch.setattr("mlca.sandbox.level.detect", lambda *_: "A")

    folder = root / "batch-ctl"
    folder.mkdir(parents=True)
    (folder / ".control").mkdir()
    (folder.parent / "batch-start.json").write_text(json.dumps({"started_at": batch.utc()}))

    state = {
        **initial_identity,
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
        "jobs": [
            {
                "id": "t-001",
                "dataset": "t",
                "replicate": 1,
                "large": False,
                "status": "finished",
                "attempts": ["studies/t-001/attempt-001"],
                "result": {"status": "success", "category": "clean", "workflow_completed": True},
            }
        ],
    }
    (folder / "batch.json").write_text(json.dumps(state))

    upgraded_identity = {
        "code_commit": "code_A",
        "controller_commit": "ctl_B",
        "controller_tree": "tree_B",
        "rails": rails_info,
    }
    monkeypatch.setattr(batch, "source_identity", lambda: upgraded_identity)
    monkeypatch.setattr(batch, "git_head", lambda _: "ctl_B")
    monkeypatch.setattr(batch, "git_tree", lambda *_: "tree_B")

    monkeypatch.setattr(sys, "argv", ["run_batch", "--out", str(folder), "--resume"])
    with pytest.raises(ValueError, match="Resume requires the original code, rails and controller version"):
        batch.main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_batch", "--out", str(folder), "--resume", "--controller-upgrade", "vm_migration"],
    )
    batch.main()
    s2 = json.loads((folder / "batch.json").read_text())
    assert s2["controller_commit"] == "ctl_B"
    assert s2["controller_tree"] == "tree_B"
    assert len(s2["controller_history"]) == 1
    assert s2["controller_history"][0]["from"] == "ctl_A"
    assert s2["controller_history"][0]["to"] == "ctl_B"
    assert s2["controller_history"][0]["reason"] == "vm_migration"

    bad_code_identity = {
        "code_commit": "code_CHANGED",
        "controller_commit": "ctl_B",
        "controller_tree": "tree_B",
        "rails": rails_info,
    }
    monkeypatch.setattr(batch, "source_identity", lambda: bad_code_identity)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_batch", "--out", str(folder), "--resume", "--controller-upgrade", "vm_migration"],
    )
    with pytest.raises(ValueError, match="Resume requires the original code commit"):
        batch.main()

    bad_rails_identity = {
        "code_commit": "code_A",
        "controller_commit": "ctl_B",
        "controller_tree": "tree_B",
        "rails": {"version": "rails-99", "sha256": "bad"},
    }
    monkeypatch.setattr(batch, "source_identity", lambda: bad_rails_identity)
    with pytest.raises(ValueError, match="Resume requires the original rails version and hash"):
        batch.main()


def test_outside_scripts_commit_resumes_without_flag(tmp_path, monkeypatch):
    code = tmp_path / "code"
    root = tmp_path / "experiments"
    (code / "rails").mkdir(parents=True)
    (code / "rails/VERSION").write_text("rails-0.3\n")
    (code / "benchmarks").mkdir()
    root.mkdir()
    data = code / "benchmarks" / "t.csv"
    data.write_text("c\n1\n")
    entries = [{
        "id": "t",
        "file": data.name,
        "role": "development",
        "rows": 1,
        "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    }]
    monkeypatch.setattr(batch, "CODE", code)
    monkeypatch.setattr(batch, "ROOT", root)
    rails_info = {"version": "rails-0.3", "sha256": load(code / "rails").sha256}
    initial_identity = {
        "code_commit": "code_A",
        "controller_commit": "ctl_A",
        "controller_tree": "tree_SAME",
        "rails": rails_info,
    }
    monkeypatch.setattr(batch, "entries", lambda *a, **k: entries)
    monkeypatch.setattr(batch, "disk_needed", lambda _: 0)
    monkeypatch.setattr(batch, "runtime_identity", lambda _: {})
    monkeypatch.setattr(batch, "frozen", lambda _: [])
    monkeypatch.setenv("OPENROUTER_API_KEY", "SYNTHETIC_KEY")
    monkeypatch.setattr("mlca.sandbox.level.detect", lambda *_: "A")

    folder = root / "batch-outside"
    folder.mkdir(parents=True)
    (folder / ".control").mkdir()
    (folder.parent / "batch-start.json").write_text(json.dumps({"started_at": batch.utc()}))

    state = {
        **initial_identity,
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
        "jobs": [
            {
                "id": "t-001",
                "dataset": "t",
                "replicate": 1,
                "large": False,
                "status": "finished",
                "attempts": ["studies/t-001/attempt-001"],
                "result": {"status": "success", "category": "clean", "workflow_completed": True},
            }
        ],
    }
    (folder / "batch.json").write_text(json.dumps(state))

    doc_commit_identity = {
        "code_commit": "code_A",
        "controller_commit": "ctl_DOCS",
        "controller_tree": "tree_SAME",
        "rails": rails_info,
    }
    monkeypatch.setattr(batch, "source_identity", lambda: doc_commit_identity)
    monkeypatch.setattr(batch, "git_head", lambda _: "ctl_DOCS")
    monkeypatch.setattr(batch, "git_tree", lambda *_: "tree_SAME")

    monkeypatch.setattr(sys, "argv", ["run_batch", "--out", str(folder), "--resume"])
    batch.main()
    s2 = json.loads((folder / "batch.json").read_text())
    assert s2["controller_commit"] == "ctl_DOCS"
    assert len(s2.get("controller_history", [])) == 1
    assert s2["controller_history"][0]["reason"] == "outside_scripts_commit"


def test_network_outage_hold_launches_nothing(tmp_path, monkeypatch):
    code = tmp_path / "code"
    root = tmp_path / "experiments"
    (code / "rails").mkdir(parents=True)
    (code / "rails/VERSION").write_text("rails-0.3\n")
    (code / "benchmarks").mkdir()
    root.mkdir()
    data = code / "benchmarks" / "t.csv"
    data.write_text("c\n1\n")
    entries = [{
        "id": "t",
        "file": data.name,
        "role": "development",
        "rows": 1,
        "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    }]
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
    monkeypatch.setattr(batch, "git_head", lambda _: "ctl")
    monkeypatch.setattr(batch, "git_tree", lambda *_: "tree")
    monkeypatch.setattr(batch, "runtime_identity", lambda _: {})
    monkeypatch.setattr(batch, "frozen", lambda _: [])
    monkeypatch.setenv("OPENROUTER_API_KEY", "SYNTHETIC_KEY")
    monkeypatch.setattr("mlca.sandbox.level.detect", lambda *_: "A")

    probe_results = [False, False, True]
    monkeypatch.setattr(
        batch, "probe_endpoint", lambda _: probe_results.pop(0) if probe_results else True
    )
    orig_sleep = time.sleep
    monkeypatch.setattr(batch.time, "sleep", lambda _: orig_sleep(0.01))

    original_popen = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        spec = json.loads(Path(command[-1]).read_text())
        p = Path(spec["attempt"])
        p.mkdir(parents=True, exist_ok=True)
        script = 'import json,sys,time;from pathlib import Path;s=json.load(open(sys.argv[1]));p=Path(s["attempt"]);(p/"worker-result.json").write_text(json.dumps({"status":"failed","category":"test_done"}))'
        proc = original_popen([sys.executable, "-c", script, command[-1]], **kwargs)
        children.append(proc)
        return proc

    monkeypatch.setattr(batch.subprocess, "Popen", launch)
    folder = root / "batch-net"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_batch", "--out", str(folder), "--batch", "1", "--model", "synthetic/model"],
    )
    batch.main()
    state = json.loads((folder / "batch.json").read_text())
    assert state["status"] == "finished"
    assert len(children) == 1
    assert "network_waits" in state
    assert len(state["network_waits"]) == 1
    assert state["network_waits"][0]["seconds"] >= 0


def test_connection_failure_filter_and_retry(tmp_path, monkeypatch):
    job_5_shaped = {
        "id": "j1",
        "result": {
            "status": "failed",
            "category": "provider_error",
            "furthest_event": "setup",
            "error_type": "MlcaUserError",
            "message": "Setup failed: MlcaUserError: Connection error.",
            "usage": {"known": {"input_tokens": 0, "output_tokens": 0}},
        },
    }
    job_mid_study = {
        "id": "j2",
        "result": {
            "status": "failed",
            "category": "provider_error",
            "furthest_event": "model_train",
            "error_type": "ConnectionError",
            "message": "Connection error mid-training",
            "usage": {"known": {"input_tokens": 500, "output_tokens": 100}},
        },
    }
    job_runtime = {
        "id": "j3",
        "result": {
            "status": "failed",
            "category": "runtime_error",
            "furthest_event": "setup",
            "error_type": "ValueError",
            "message": "Invalid argument",
            "usage": {"known": {"input_tokens": 0, "output_tokens": 0}},
        },
    }
    assert batch.is_connection_failure(job_5_shaped) is True
    assert batch.is_connection_failure(job_mid_study) is False
    assert batch.is_connection_failure(job_runtime) is False

    code = tmp_path / "code"
    root = tmp_path / "experiments"
    (code / "rails").mkdir(parents=True)
    (code / "rails/VERSION").write_text("rails-0.3\n")
    (code / "benchmarks").mkdir()
    root.mkdir()
    data = code / "benchmarks" / "t.csv"
    data.write_text("c\n1\n")
    entries = [{
        "id": "t",
        "file": data.name,
        "role": "development",
        "rows": 1,
        "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    }]
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
    monkeypatch.setattr(batch, "git_head", lambda _: "ctl")
    monkeypatch.setattr(batch, "git_tree", lambda *_: "tree")
    monkeypatch.setattr(batch, "runtime_identity", lambda _: {})
    monkeypatch.setattr(batch, "frozen", lambda _: [])
    monkeypatch.setenv("OPENROUTER_API_KEY", "SYNTHETIC_KEY")
    monkeypatch.setattr("mlca.sandbox.level.detect", lambda *_: "A")

    folder = root / "batch-retry"
    folder.mkdir(parents=True)
    (folder / ".control").mkdir()
    (folder.parent / "batch-start.json").write_text(json.dumps({"started_at": batch.utc()}))

    job_to_retry = {
        "id": "t-001",
        "dataset": "t",
        "replicate": 1,
        "large": False,
        "status": "finished",
        "first_started": 12345.0,
        "attempts": ["studies/t-001/attempt-001"],
        "result": {
            "status": "failed",
            "category": "provider_error",
            "furthest_event": "setup",
            "error_type": "MlcaUserError",
            "message": "Setup failed: MlcaUserError: Connection error.",
            "usage": {"known": {"input_tokens": 0, "output_tokens": 0}},
        },
    }
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
        "jobs": [job_to_retry],
    }
    (folder / "batch.json").write_text(json.dumps(state))

    original_popen = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        spec = json.loads(Path(command[-1]).read_text())
        p = Path(spec["attempt"])
        p.mkdir(parents=True, exist_ok=True)
        script = 'import json,sys,time;from pathlib import Path;s=json.load(open(sys.argv[1]));p=Path(s["attempt"]);(p/"worker-result.json").write_text(json.dumps({"status":"success","category":"clean","workflow_completed":True}))'
        proc = original_popen([sys.executable, "-c", script, command[-1]], **kwargs)
        children.append(proc)
        return proc

    monkeypatch.setattr(batch.subprocess, "Popen", launch)

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_batch", "--out", str(folder), "--resume", "--retry-connection-failures"],
    )
    batch.main()
    s2 = json.loads((folder / "batch.json").read_text())
    j = s2["jobs"][0]
    assert j["status"] == "finished"
    assert len(j.get("retries", [])) == 1
    assert j["retries"][0]["retry_of"] == "studies/t-001/attempt-001"
    assert j["retries"][0]["reason"] == "connection_failure"
    assert j["first_started_original"] == 12345.0
    assert len(j["attempts"]) == 2
    assert j["attempts"][-1] == "studies/t-001/attempt-002"
    assert len(children) == 1

    # Second retry of the same job is refused
    j["result"] = {
        "status": "failed",
        "category": "provider_error",
        "furthest_event": "setup",
        "error_type": "MlcaUserError",
        "message": "Connection error again",
        "usage": {"known": {"input_tokens": 0, "output_tokens": 0}},
    }
    (folder / "batch.json").write_text(json.dumps(s2))
    with pytest.raises(ValueError, match="already retried once; refusing second retry"):
        batch.main()


def test_relocated_folder_resumes(tmp_path, monkeypatch):
    import shutil

    code = tmp_path / "code"
    root = tmp_path / "experiments"
    (code / "rails").mkdir(parents=True)
    (code / "rails/VERSION").write_text("rails-0.3\n")
    (code / "benchmarks").mkdir()
    root.mkdir()
    data = code / "benchmarks" / "t.csv"
    data.write_text("c\n1\n")
    entries = [{
        "id": "t",
        "file": data.name,
        "role": "development",
        "rows": 1,
        "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    }]
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
    monkeypatch.setattr(batch, "git_head", lambda _: "ctl")
    monkeypatch.setattr(batch, "git_tree", lambda *_: "tree")
    monkeypatch.setattr(batch, "runtime_identity", lambda _: {})
    monkeypatch.setattr(batch, "frozen", lambda _: [])
    monkeypatch.setenv("OPENROUTER_API_KEY", "SYNTHETIC_KEY")
    monkeypatch.setattr("mlca.sandbox.level.detect", lambda *_: "A")

    folder_orig = root / "batch-orig"
    folder_orig.mkdir(parents=True)
    (folder_orig / ".control").mkdir()
    (folder_orig.parent / "batch-start.json").write_text(json.dumps({"started_at": batch.utc()}))

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
        "rate_limit_dir": str(folder_orig / ".control/rate"),
        "machine": {"cpus": 2},
        "jobs": [
            {
                "id": "t-001",
                "dataset": "t",
                "replicate": 1,
                "large": False,
                "status": "finished",
                "attempts": ["studies/t-001/attempt-001"],
                "result": {"status": "success", "category": "clean", "workflow_completed": True},
            }
        ],
    }
    (folder_orig / "batch.json").write_text(json.dumps(state))

    folder_moved = root / "batch-moved"
    shutil.copytree(folder_orig, folder_moved)

    monkeypatch.setattr(sys, "argv", ["run_batch", "--out", str(folder_moved), "--resume"])
    batch.main()
    s_moved = json.loads((folder_moved / "batch.json").read_text())
    assert s_moved["rate_limit_dir"] == str(folder_moved / ".control/rate")

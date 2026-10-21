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
        + (["--tables", "table-0"] if subset else [])
        + (
            ["--sdk-timeout-s", "300", "--model-call-deadline-s", "600"]
            if provider == "ollama"
            else []
        ),
    )
    batch.main()
    state = json.loads((folder / "batch.json").read_text())
    expected = (1 if subset else 12) * replicates
    assert len(children) == expected and max(peaks) == (1 if subset else 6)
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

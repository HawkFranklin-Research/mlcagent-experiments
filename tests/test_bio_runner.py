"""The batch controller in evaluation mode: approved tasks, written briefs, verified data,
sealed external cohorts, no development gates."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import run_batch as batch
from mlca.rails import load


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    code, root, data = tmp_path / "code", tmp_path / "experiments", tmp_path / "hf"
    (code / "rails").mkdir(parents=True)
    (code / "rails/VERSION").write_text("rails-0.3\n")
    (root / "bio-bench/briefs").mkdir(parents=True)
    (data / "prepared/S3").mkdir(parents=True)
    (data / "prepared/S3/discovery.csv").write_text("g1,PAM50\n1,A\n")
    (data / "prepared/S3/external.csv").write_text("g1,PAM50\n2,B\n")
    (root / "bio-bench/briefs/s3.md").write_text("Predict the PAM50 subtype.\n")
    entry = {
        "id": "s3",
        "role": "evaluation",
        "file": "prepared/S3/discovery.csv",
        "sha256": digest(data / "prepared/S3/discovery.csv"),
        "external": "prepared/S3/external.csv",
        "external_sha256": digest(data / "prepared/S3/external.csv"),
        "rows": 1,
        "problem_type": "multiclass",
        "targets": ["PAM50"],
        "brief_file": "bio-bench/briefs/s3.md",
        "approved": True,
    }
    monkeypatch.setattr(batch, "CODE", code)
    monkeypatch.setattr(batch, "ROOT", root)
    return code, root, data, entry


def test_evaluation_requires_approval_brief_and_unchanged_data(setup):
    _code, root, data, entry = setup
    batch.validate_evaluation([entry], data, allow_draft=False)
    with pytest.raises(ValueError, match="--data-root"):
        batch.validate_evaluation([entry], None, allow_draft=False)
    with pytest.raises(ValueError, match="not approved"):
        batch.validate_evaluation(
            [{**entry, "approved": False}], data, allow_draft=False
        )
    batch.validate_evaluation([{**entry, "approved": False}], data, allow_draft=True)
    with pytest.raises(ValueError, match="Problem type undecided"):
        batch.validate_evaluation(
            [{**entry, "problem_type": "undecided"}], data, allow_draft=True
        )
    (data / "prepared/S3/external.csv").write_text("changed\n")
    with pytest.raises(ValueError, match="missing or changed"):
        batch.validate_evaluation([entry], data, allow_draft=False)
    (root / "bio-bench/briefs/s3.md").write_text(" \n")
    with pytest.raises(ValueError, match="brief missing"):
        batch.validate_evaluation([entry], data, allow_draft=True)


def test_roles_select_their_tables(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "- {id: a, role: development}\n- {id: b, role: heldout}\n- {id: c, role: evaluation}\n"
    )
    assert [e["id"] for e in batch.entries(manifest, "heldout")] == ["b"]
    assert [e["id"] for e in batch.entries(manifest, "evaluation")] == ["c"]
    with pytest.raises(ValueError):
        batch.entries(manifest, "other")


@pytest.mark.parametrize(
    "provider,structured", [("openrouter", "on"), ("ollama", "off")]
)
def test_evaluation_batch_passes_brief_and_sealed_cohort_to_each_run(
    setup, monkeypatch, provider, structured
):
    code, root, data, entry = setup
    identity = {
        "code_commit": "fixture",
        "controller_commit": "fixture",
        "rails": {"version": "rails-0.3", "sha256": load(code / "rails").sha256},
    }
    monkeypatch.setattr(batch, "source_identity", lambda: identity)
    monkeypatch.setattr(batch, "entries", lambda *a, **k: [entry])
    monkeypatch.setattr(batch, "frozen", lambda _: ["freeze-test"])
    monkeypatch.setattr(batch, "disk_needed", lambda job: 0)
    monkeypatch.setattr(batch, "git_head", lambda _: "fixture")
    monkeypatch.setattr(batch, "runtime_identity", lambda _: {})
    monkeypatch.setenv(batch.PROVIDERS[provider][1], "SYNTHETIC_NO_NETWORK")
    monkeypatch.setattr("mlca.sandbox.level.detect", lambda *_: "A")
    specs = []
    original_popen = subprocess.Popen

    def launch(command, **kwargs):
        spec = json.loads(Path(command[-1]).read_text())
        specs.append(spec)
        script = 'import json,sys;from pathlib import Path;s=json.load(open(sys.argv[1]));p=Path(s["attempt"]);p.mkdir(parents=True);(p/"worker-result.json").write_text(json.dumps({"status":"failed","category":"synthetic"}))'
        return original_popen([sys.executable, "-c", script, command[-1]], **kwargs)

    monkeypatch.setattr(batch.subprocess, "Popen", launch)
    original_sleep = time.sleep
    monkeypatch.setattr(batch.time, "sleep", lambda _: original_sleep(0.03))
    folder = root / "bio-bench/runs/eval-01"
    argv = [
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
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        argv + ["--role", "evaluation", "--data-root", str(data), "--replicates", "2"],
    )
    batch.main()
    state = json.loads((folder / "batch.json").read_text())
    assert state["stage"] == "evaluation" and state["role"] == "evaluation"
    assert len(specs) == 2
    for spec in specs:
        config = spec["config"]["providers"]["development"]
        assert config["kind"] == (
            "openai_compatible" if provider == "ollama" else provider
        )
        assert config["base_url"] == batch.PROVIDERS[provider][0]
        assert config["api_key_env"] == batch.PROVIDERS[provider][1]
        assert config["structured_outputs"] is (structured == "on")
    assert all(s["brief"] == "Predict the PAM50 subtype." for s in specs)
    assert all(s["external"] == str(data / "prepared/S3/external.csv") for s in specs)
    assert all(s["data"] == str(data / "prepared/S3/discovery.csv") for s in specs)
    assert all(j["status"] == "finished" for j in state["jobs"])


def test_shipped_evaluation_manifest_is_complete_and_approved():
    import yaml

    tasks = yaml.safe_load(
        (Path(__file__).parents[1] / "bio-bench/manifest.yaml").read_text()
    )
    assert len({t["id"] for t in tasks}) == len(tasks) == 9
    for task in tasks:
        assert task["role"] == "evaluation" and task["approved"] is True
        if task["problem_type"] == "survival":
            assert task["duration"] and task["event_column"]
        assert task["brief_file"] == f"bio-bench/briefs/{task['id']}.md"
        assert len(task["sha256"]) == 64
        assert bool(task.get("external")) == bool(task.get("external_sha256"))

"""Development orchestration only: no analysis code, no held-out inputs, no secrets on disk."""

import argparse
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT.parent / "mlca-revized"
CONFIG = {
    "measurement_arm": "worker",
    "providers": {
        "free_router": {
            "kind": "openrouter",
            "api_key_env": "OPENROUTER_API_KEY",
            "requests_per_minute": 16,
            "request_cost_bound_usd": 0,
            "timeout_s": 90,
        }
    },
    "roles": {
        role: {
            "provider": "free_router",
            "model": "openrouter/free",
            "temperature": 0,
            "max_output_tokens": 16384,
        }
        for role in ["navigator", "worker"]
    },
    "sandbox": {"allow_level_b": False},
}


def quota():
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.load(response).get("data", {})
    daily = data.get("free_model_daily_requests")
    if not isinstance(daily, dict) or not all(
        type(daily.get(k)) is int and daily[k] >= 0
        for k in ["used", "limit", "remaining"]
    ):
        raise RuntimeError("Free-request allowance unavailable; do not invent a quota")
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "source": "OpenRouter GET /api/v1/key",
        "free_model_daily_requests": {
            k: daily[k] for k in ["used", "limit", "remaining"]
        },
    }


def development_entries(manifest, only=()):
    entries = [e for e in manifest if e["role"] == "development"]
    if len(entries) != 12:
        raise ValueError("Expected the author-approved twelve development tables")
    ranks = {"simple": 0, "moderate": 1, "medium": 1, "complex": 2}
    entries.sort(key=lambda e: ranks.get(e["level"], 2))
    if only:
        if not set(only) <= {e["id"] for e in entries}:
            raise ValueError("Only development IDs may be selected")
        entries = [e for e in entries if e["id"] in only]
    return entries


def write_summary(folder, state):
    from mlca.paths import atomic_json

    atomic_json(folder / "round.json", state)
    lines = [
        "# Rails development round",
        "",
        "Live Worker-written-code development; not frozen scientific evaluation.",
        "",
        "| Table | Status | Requests | Failure category | Classification |",
        "|---|---|---:|---|---|",
    ]
    lines += [
        f"| {r['dataset']} | {r['status']} | {r.get('requests')} | {r.get('category', '')} | {r.get('classification', '')} |"
        for r in state["runs"]
    ]
    if state.get("stop_reason"):
        lines += ["", "Stop: " + state["stop_reason"]]
    (folder / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True, choices=range(1, 13))
    parser.add_argument("--runs", type=int, default=1, choices=[1, 3])
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Provide OPENROUTER_API_KEY in the environment")
    # Isolate provider selection from the user's global configuration. Credentials
    # remain only in the environment and analysis subprocesses use MLCA's scrubber.
    clean_config = Path.home() / ".cache/mlca-dev-loop/config"
    clean_config.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="config-", dir=clean_config)
    os.environ["UV_OFFLINE"] = "1"
    from mlca.autonomous.runner import run
    from mlca.evaluation.export import export
    from mlca.llm.usage import read, summary
    from mlca.paths import atomic_json
    from mlca.rails import load
    from score_inference import score

    manifest = yaml.safe_load((CODE / "benchmarks/manifest.yaml").read_text())
    entries = development_entries(manifest, args.only)
    folder = ROOT / "bench/dev-loop" / f"round-{args.round:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    started_file = ROOT / "bench/dev-loop/started.json"
    if not started_file.exists():
        atomic_json(
            started_file, {"started_at": datetime.now(timezone.utc).isoformat()}
        )
    started_at = datetime.fromisoformat(
        json.loads(started_file.read_text())["started_at"]
    )
    baseline = load(CODE / "rails")
    state = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "round": args.round,
        "code_commit": subprocess.check_output(
            ["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True
        ).strip(),
        "rails": {"version": baseline.version, "sha256": baseline.sha256},
        "config": CONFIG,
        "runs_per_table": args.runs,
        "runs": [],
        "quota": [],
    }
    write_summary(folder, state)
    observed = []
    for i, entry in enumerate(entries):
        current_commit = subprocess.check_output(
            ["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(CODE),
                "diff",
                "--quiet",
                "HEAD",
                "--",
                "src",
                "templates",
                "rails",
                "pyproject.toml",
                "uv.lock",
            ],
            check=False,
        ).returncode
        if current_commit != state["code_commit"] or dirty:
            state["stop_reason"] = (
                "Code changed during the round; do not mix implementations"
            )
            break
        if load(CODE / "rails").sha256 != baseline.sha256:
            state["stop_reason"] = (
                "Rails changed during a round; do not mix implementations"
            )
            break
        if (datetime.now(timezone.utc) - started_at).total_seconds() >= 5 * 86400:
            state["stop_reason"] = "Five-day development cap reached"
            break
        try:
            checked = quota()
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            state["stop_reason"] = "Required quota check failed: " + type(exc).__name__
            break
        state["quota"].append(checked)
        estimate = max(50, sum(observed) / len(observed)) if observed else 50
        needed = int(estimate * args.runs * (len(entries) - i))
        if checked["free_model_daily_requests"]["remaining"] < needed:
            state["stop_reason"] = (
                f"Free daily quota cannot carry remaining round: estimate {needed}, remaining {checked['free_model_daily_requests']['remaining']}"
            )
            break
        started = time.monotonic()
        # Manifest targets/traps/task labels never enter this call. Only excluded
        # evaluation-column names cross the explicit author-approved interface.
        result = run(
            CODE / "benchmarks" / entry["file"],
            folder / entry["id"],
            runs=args.runs,
            config=CONFIG,
            exclude_columns=entry.get("evaluation_only", []),
            halt_window=0,
        )
        for item in result["runs"]:
            project = folder / entry["id"] / item["project"]
            setup = folder / entry["id"] / f"setup-{item['run']:03d}"
            requests = read(project) if project.exists() else read(setup)
            usage = summary(requests) if project.exists() else item.get("setup_usage")
            number = usage.get("model_requests") if usage else None
            if number is not None:
                observed.append(number)
            row = {
                "dataset": entry["id"],
                **item,
                "project": str(project.relative_to(folder)),
                "requests": number,
                "wall_s_table": time.monotonic() - started,
                "classification": "provider"
                if item.get("category") == "provider_error"
                else "pending_review"
                if item["status"] != "completed"
                else "completed",
                "model_reported": dict(
                    Counter(
                        r["model_reported"] for r in requests if r.get("model_reported")
                    )
                ),
            }
            state["runs"].append(row)
            if project.exists():
                from mlca.log.store import Store

                store = Store(project)
                store.verify()
                proposals = [
                    r.payload["proposal"]
                    for r in store.read()
                    if r.type.value == "InferenceRecorded"
                ]
                if proposals:
                    atomic_json(
                        project.parent / (project.name + "-inference-score.json"),
                        score(proposals[-1], entry),
                    )
            if item["status"] == "completed" and entry.get("evaluation_only"):
                expected_kind = entry["problem_type"]
                output_name = {
                    "clustering": "assignments.parquet",
                    "representation": "embeddings.parquet",
                }.get(expected_kind)
                outputs = (
                    list((project / ".mlca/steps").rglob(output_name))
                    if output_name
                    else []
                )
                if len(outputs) == 1:
                    from score_unsupervised import score as score_hidden

                    try:
                        evaluated = score_hidden(
                            CODE / "benchmarks" / entry["file"],
                            outputs[0],
                            entry["evaluation_only"],
                            expected_kind,
                        )
                    except (ValueError, KeyError, TypeError) as exc:
                        evaluated = {
                            "agent_work": False,
                            "status": "not_achieved",
                            "error_type": type(exc).__name__,
                        }
                else:
                    evaluated = {
                        "agent_work": False,
                        "status": "not_achieved",
                        "reason": "No unique required per-row output",
                    }
                atomic_json(
                    project.parent / (project.name + "-external-evaluation.json"),
                    evaluated,
                )
            print(
                json.dumps(
                    {
                        "dataset": row["dataset"],
                        "run": row["run"],
                        "status": row["status"],
                        "requests": number,
                    }
                ),
                flush=True,
            )
        export(folder / entry["id"], folder / "curated" / entry["id"])
        write_summary(folder, state)
    state["ended_at"] = datetime.now(timezone.utc).isoformat()
    write_summary(folder, state)


if __name__ == "__main__":
    main()

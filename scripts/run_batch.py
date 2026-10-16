"""Resumable process queue for author-approved development batches; never analysis code."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT.parent / "mlca-revized"
sys.path.insert(0, str(CODE / "src"))
from mlca.llm.usage import summary
from mlca.paths import atomic_json
from mlca.rails import load

TERMINAL = {"finished"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def git_head(root):
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def source_identity():
    for root in [CODE, ROOT]:
        if subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD"], check=False
        ).returncode:
            raise ValueError(
                "Tracked working tree must be clean before starting/resuming a batch"
            )
    for root, paths in [(CODE, ["src", "templates", "rails"]), (ROOT, ["scripts"])]:
        unknown = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                *paths,
            ],
            text=True,
        )
        if unknown.strip():
            raise ValueError("Untracked runtime source is not reproducible")
    rails = load(CODE / "rails")
    return {
        "code_commit": git_head(CODE),
        "controller_commit": git_head(ROOT),
        "rails": {"version": rails.version, "sha256": rails.sha256},
    }


def validate_stage(folder, identity, model, selected, all_ids, replicates, rails_off, reasoning_effort=None):
    """Require recorded successes before acceptance, and acceptance before ablation."""
    prior = []
    for path in sorted(folder.parent.glob("batch-[0-9][0-9]/batch.json")):
        if path.parent == folder:
            continue
        state = json.loads(path.read_text())
        if state.get("status") == "finished" and state.get("model") == model:
            prior.append(state)
    if replicates == 3:
        succeeded = {
            job["dataset"]
            for state in prior
            if state.get("rails_enabled")
            for job in state["jobs"]
            if job.get("result", {}).get("success")
        }
        if set(selected) != all_ids or not all_ids <= succeeded or rails_off:
            raise ValueError(
                "Acceptance requires all development tables to have succeeded once"
            )
    if rails_off:
        accepted = any(
            state.get("replicates") == 3
            and state.get("rails_enabled")
            and state.get("code_commit") == identity["code_commit"]
            and state.get("rails") == identity["rails"]
            and state.get("reasoning_effort") == reasoning_effort
            and len(state["jobs"]) == 3 * len(all_ids)
            and all(j.get("result", {}).get("success") for j in state["jobs"])
            for state in prior
        )
        if not accepted or set(selected) != all_ids or replicates != 1:
            raise ValueError(
                "Ablation requires successful K=3 acceptance on the same version"
            )


def entries():
    return [
        e
        for e in yaml.safe_load((CODE / "benchmarks/manifest.yaml").read_text())
        if e["role"] == "development"
    ]


def choose_next(jobs, active, concurrency):
    slots = concurrency - len(active)
    large_running = any(j["large"] for j in active)
    chosen = []
    for job in jobs:
        if slots <= 0:
            break
        if job["status"] != "queued" or job["large"] and large_running:
            continue
        chosen.append(job)
        slots -= 1
        large_running |= job["large"]
    return chosen


def request_records(folder):
    records = {}
    for file in Path(folder).glob("studies/*/attempt-*/**/.mlca/requests/*.json"):
        try:
            record = json.loads(file.read_text())["usage"]
        except (OSError, ValueError, KeyError):
            continue
        if record.get("role") == "mcp":
            continue
        previous = records.get(record["request_id"])
        rank = lambda r: (
            r.get("status") != "pending",
            sum(
                r.get(k) is not None
                for k in ["input_tokens", "output_tokens", "cost_usd"]
            ),
        )
        if previous is None or rank(record) > rank(previous):
            records[record["request_id"]] = record
    return list(records.values())


def accounting(records):
    totals = summary(records)["totals"]
    return {
        "requests": len(records),
        "known": {
            k: totals.get(k)
            for k in ["input_tokens", "output_tokens", "reasoning_tokens", "cost_usd"]
        },
        "unknown": {
            k: totals["unknown_" + k + "_requests"]
            for k in ["input_tokens", "output_tokens", "cost_usd"]
        },
        "partial": totals["partial"],
        "models": dict(
            Counter(r.get("model_reported") or "unavailable" for r in records)
        ),
    }


def alive(job):
    pid = job.get("pid")
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] != "Z" and fields[19] == job.get("process_start")
    except (OSError, IndexError):
        return False


def start_token(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def finish_job(folder, job, expected):
    attempt = folder / job["attempts"][-1]
    result_file = attempt / "worker-result.json"
    result = (
        json.loads(result_file.read_text())
        if result_file.exists()
        else {"status": "failed", "category": "worker_process_failure"}
    )
    if job.get("stop_reason"):
        result = {"status": "failed", "category": job["stop_reason"]}
    project = attempt / "run-001"
    request_list = {}
    for relative in job["attempts"]:
        for root in [folder / relative / "setup-001", folder / relative / "run-001"]:
            for file in (root / ".mlca/requests").glob("*.json"):
                r = json.loads(file.read_text())["usage"]
                request_list[r["request_id"]] = r
    result["usage"] = accounting(list(request_list.values()))
    result["repair_calls"] = sum(
        r.get("purpose") == "repair" for r in request_list.values()
    )
    result["workflow_completed"] = result.get("status") == "completed"
    job["result"] = result
    result["setup_agreement"] = None
    result["leakage"] = {"leakage_clean": None, "criteria": {}}
    events = project / ".mlca/log/events.jsonl"
    result["furthest_event"] = "setup"
    result["furthest_step"] = None
    if events.exists():
        from mlca.log.store import Store

        store = Store(project)
        store.verify()
        records = store.read()
        for record in records:
            if record.event_id:
                result["furthest_event"] = record.event_id
            if record.step_id:
                result["furthest_step"] = record.step_id
        proposals = [
            r.payload["proposal"]
            for r in records
            if r.type.value == "InferenceRecorded"
        ]
        if proposals:
            from score_inference import score

            scored = score(proposals[-1], expected)
            atomic_json(attempt / "setup-agreement.json", scored)
            result["setup_agreement"] = all(
                scored[k]
                for k in ["problem_type", "targets", "group_column", "time_columns"]
            )
            result["setup_details"] = scored
        from score_leakage import score as leakage_score

        result["leakage"] = leakage_score(project, expected)
        atomic_json(attempt / "leakage-score.json", result["leakage"])
        attempts = {
            (r.step_id, r.payload["attempt"])
            for r in records
            if r.type.value == "ScriptRequested"
        }
        result["script_attempts"] = len(attempts)
    result["success"] = bool(
        result["workflow_completed"]
        and result["setup_agreement"] is True
        and result["leakage"]["leakage_clean"] is True
    )
    if result["workflow_completed"] and not result["success"]:
        result["category"] = (
            "setup_disagreement"
            if result["setup_agreement"] is False
            else "leakage_failed"
            if result["leakage"]["leakage_clean"] is False
            else "evaluation_unavailable"
        )
    result["wall_s"] = max(0, time.time() - job["first_started"])
    job.update(
        status="interrupted"
        if job.get("stop_reason") == "interrupted_by_author"
        else "finished",
        finished_at=utc(),
        result=result,
    )


def record_resources(folder, state):
    now = time.time()
    if now - state.get("last_resource_sample", 0) < 30:
        return
    sample = {"ts": utc(), "disk_free_bytes": shutil.disk_usage(folder).free}
    if Path("/proc/meminfo").is_file():
        memory = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemAvailable", "SwapFree", "SwapTotal"}:
                memory[key + "_kib"] = int(value.split()[0])
        sample.update(memory)
    pressure = Path("/proc/pressure/memory")
    if pressure.is_file():
        sample["memory_pressure"] = pressure.read_text().strip()
    sample["active_studies"] = sum(
        j["status"] in {"running", "stopping"} for j in state["jobs"]
    )
    with (folder / "resources.jsonl").open("a") as stream:
        stream.write(json.dumps(sample) + "\n")
    state["last_resource_sample"] = now


def write_report(folder, state):
    record_resources(folder, state)
    costs = accounting(request_records(folder))
    atomic_json(folder / "cost-ledger.json", costs)
    cumulative = []
    for path in sorted(folder.parent.glob("batch-[0-9][0-9]")):
        cumulative.extend(request_records(path))
    running = accounting(list({r["request_id"]: r for r in cumulative}.values()))
    state["accounting"] = costs
    state["running_accounting"] = running
    atomic_json(folder / "batch.json", state)
    lines = [
        "# Development batch " + str(state["batch"]),
        "",
        "Development only; model, runtime source and rails fixed for this batch. No comparison with exploratory rounds.",
        "",
        "| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |",
        "|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|",
    ]
    for job in state["jobs"]:
        r = job.get("result", {})
        u = r.get("usage", {}).get("known", {})
        criteria = r.get("leakage", {}).get("criteria", {})
        flags = " / ".join(
            str(criteria.get(k, {}).get("value"))
            for k in ["fitted_columns", "group_disjoint", "time_ordered"]
        )
        lines.append(
            f"| {job['dataset']} / {job['replicate']} | {r.get('workflow_completed', job['status'])} | {r.get('setup_agreement')} | {flags} | {r.get('success')} | {r.get('furthest_event', 'setup')} / {r.get('furthest_step')} | {r.get('category', '')} | {r.get('repair_calls')} | {r.get('usage', {}).get('requests')} | {u.get('input_tokens')} / {u.get('output_tokens')} / {u.get('reasoning_tokens')} | {u.get('cost_usd')} | {r.get('wall_s')} | {json.dumps(r.get('usage', {}).get('models', {}))} |"
        )
    lines += [
        "",
        "## Batch accounting",
        "",
        "```json",
        json.dumps(costs, indent=2),
        "```",
        "",
        "## Running accounting since batch-01",
        "",
        "```json",
        json.dumps(running, indent=2),
        "```",
        "",
        "Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.",
    ]
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n")


def worker(spec_file):
    spec = json.loads(Path(spec_file).read_text())
    attempt = Path(spec["attempt"])
    os.environ.update(spec["environment"])
    if hasattr(os, "sched_setaffinity") and spec.get("affinity"):
        os.sched_setaffinity(0, spec["affinity"])
    from mlca.autonomous.runner import run

    try:
        result = run(
            Path(spec["data"]),
            attempt,
            config=spec["config"],
            exclude_columns=spec["exclude_columns"],
            halt_window=0,
        )
        result = result["runs"][0]
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - isolate one study, retaining its outcome
        result = {
            "status": "failed",
            "category": "interrupted"
            if isinstance(exc, KeyboardInterrupt)
            else "worker_exception",
            "error_type": type(exc).__name__,
        }
    atomic_json(attempt / "worker-result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=range(1, 11))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "max"])
    parser.add_argument("--concurrency", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--deadline-s", type=float, default=5400)
    parser.add_argument("--rpm", type=float, default=60)
    parser.add_argument("--tables", nargs="*")
    parser.add_argument("--replicates", type=int, choices=[1, 3], default=1)
    parser.add_argument("--rails-off", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.deadline_s <= 0 or args.rpm <= 0:
        raise ValueError("Deadlines and request rates must be positive")
    identity = source_identity()
    all_entries = entries()
    by_id = {e["id"]: e for e in all_entries}
    folder = args.out.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ".control").mkdir(exist_ok=True)
    interrupted = [False]
    signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))
    with FileLock(str(folder / ".control/controller.lock"), timeout=0):
        state_file = folder / "batch.json"
        if args.resume:
            state = json.loads(state_file.read_text())
            if any(state[k] != identity[k] for k in identity):
                raise ValueError(
                    "Resume requires the original code, rails and controller version"
                )
            for job in state["jobs"]:
                if job["status"] == "interrupted":
                    job["status"] = "queued"
                    job.pop("stop_reason", None)
                if job["status"] in {"running", "stopping"} and not alive(job):
                    attempt = folder / job["attempts"][-1]
                    if (attempt / "worker-result.json").exists():
                        finish_job(folder, job, by_id[job["dataset"]])
                    else:
                        job["status"] = "queued"
                        job.pop("stop_reason", None)
        else:
            if state_file.exists() or not args.batch or not args.model:
                raise ValueError(
                    "New batch needs a fresh directory, --batch and --model"
                )
            selected = args.tables or list(by_id)
            if (
                not set(selected) <= set(by_id)
                or args.batch == 1
                and set(selected) != set(by_id)
            ):
                raise ValueError(
                    "Only development tables; batch-01 must include all twelve"
                )
            validate_stage(
                folder,
                identity,
                args.model,
                selected,
                set(by_id),
                args.replicates,
                args.rails_off,
                args.reasoning_effort,
            )
            started = folder.parent / "batch-start.json"
            if not started.exists():
                atomic_json(started, {"started_at": utc()})
            state = {
                **identity,
                "batch": args.batch,
                "started_at": utc(),
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "concurrency": args.concurrency,
                "deadline_s": args.deadline_s,
                "rpm": args.rpm,
                "rails_enabled": not args.rails_off,
                "replicates": args.replicates,
                "threads": max(1, len(os.sched_getaffinity(0)) // args.concurrency)
                if hasattr(os, "sched_getaffinity")
                else max(1, (os.cpu_count() or 1) // args.concurrency),
                "memory_limit": "not_enforced",
                "sandbox": "A",
                "spending_ceiling_usd": None,
                "model_call_deadline_s": 300,
                "structured_outputs": True,
                "script_timeout_s": 300,
                "max_worker_attempts": 3,
                "controller_sampling": False,
                "machine": {
                    "logical_cpus": os.cpu_count(),
                    "platform": sys.platform,
                    "python": sys.version,
                },
                "jobs": [
                    {
                        "id": f"{name}-{rep:03d}",
                        "dataset": name,
                        "replicate": rep,
                        "large": by_id[name]["rows"] > 500000,
                        "status": "queued",
                        "attempts": [],
                    }
                    for name in selected
                    for rep in range(1, args.replicates + 1)
                ],
            }
        if Path("/proc/meminfo").is_file():
            state["machine"]["memory_total_kib"] = int(
                Path("/proc/meminfo").read_text().split("MemTotal:")[1].split()[0]
            )
        from mlca.sandbox.level import detect

        if detect(False) != "A":
            raise ValueError("Sandbox A required")
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError("Provide OPENROUTER_API_KEY in the environment")
        processes = {}
        while any(j["status"] not in TERMINAL for j in state["jobs"]):
            try:
                unchanged = source_identity() == identity
            except ValueError:
                unchanged = False
            if not unchanged:
                state["stop_reason"] = "source_changed_during_batch"
                interrupted[0] = True
            started = json.loads((folder.parent / "batch-start.json").read_text())[
                "started_at"
            ]
            if (
                datetime.now(timezone.utc) - datetime.fromisoformat(started)
            ).total_seconds() > 5 * 86400:
                state["stop_reason"] = "five_day_cap"
                interrupted[0] = True
            for job in state["jobs"]:
                if job["status"] not in {"running", "stopping"}:
                    continue
                proc = processes.get(job["id"])
                exited = (proc.poll() is not None) if proc else not alive(job)
                if exited:
                    try:
                        finish_job(folder, job, by_id[job["dataset"]])
                    except Exception as exc:  # noqa: BLE001 - one scorer failure must not stop other studies
                        job.update(
                            status="finished",
                            result={
                                **job.get("result", {}),
                                "status": "failed",
                                "success": False,
                                "category": "external_scoring_error",
                                "error_type": type(exc).__name__,
                            },
                        )
                    processes.pop(job["id"], None)
                    print(
                        json.dumps(
                            {
                                "job": job["id"],
                                "result": job["result"].get(
                                    "category", job["result"].get("status")
                                ),
                            }
                        ),
                        flush=True,
                    )
                    continue
                expired = time.time() - job["first_started"] > state["deadline_s"]
                if job["status"] == "running" and (expired or interrupted[0]):
                    job["stop_reason"] = (
                        "study_deadline" if expired else "interrupted_by_author"
                    )
                    job["status"] = "stopping"
                    job["stopping_at"] = time.time()
                    try:
                        os.killpg(job["pid"], signal.SIGINT)
                    except ProcessLookupError:
                        pass
                elif (
                    job["status"] == "stopping" and time.time() - job["stopping_at"] > 5
                ):
                    try:
                        os.killpg(job["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            active = [
                j for j in state["jobs"] if j["status"] in {"running", "stopping"}
            ]
            if interrupted[0] and not active:
                state["status"] = "interrupted"
                write_report(folder, state)
                return
            if not interrupted[0]:
                for job in choose_next(state["jobs"], active, state["concurrency"]):
                    if (
                        job.get("first_started")
                        and time.time() - job["first_started"] > state["deadline_s"]
                    ):
                        job["stop_reason"] = "study_deadline"
                        finish_job(folder, job, by_id[job["dataset"]])
                        continue
                    entry = by_id[job["dataset"]]
                    if (
                        hashlib.sha256(
                            (CODE / "benchmarks" / entry["file"]).read_bytes()
                        ).hexdigest()
                        != entry["sha256"]
                    ):
                        raise ValueError(
                            "Development source hash differs from manifest"
                        )
                    number = len(job["attempts"]) + 1
                    relative = Path("studies") / job["id"] / f"attempt-{number:03d}"
                    attempt = folder / relative
                    attempt.parent.mkdir(parents=True, exist_ok=True)
                    job["attempts"].append(str(relative))
                    job.setdefault("first_started", time.time())
                    cfg = {
                        "measurement_arm": "worker",
                        "rails_enabled": state["rails_enabled"],
                        "providers": {
                            "development": {
                                "kind": "openrouter",
                                "api_key_env": "OPENROUTER_API_KEY",
                                "timeout_s": 90,
                                "structured_outputs": state["structured_outputs"],
                                "call_deadline_s": 300,
                                "requests_per_minute": state["rpm"],
                                "rate_limit_dir": str(folder / ".control/rate"),
                            }
                        },
                        "roles": {
                            role: {
                                "provider": "development",
                                "model": state["model"],
                                "temperature": 0,
                                "max_output_tokens": 16384,
                                **({"reasoning_effort": state["reasoning_effort"]} if state.get("reasoning_effort") else {}),
                            }
                            for role in ["navigator", "worker"]
                        },
                        "budgets": {
                            "turn_max_usd": None,
                            "session_max_usd": None,
                            "worker_max_attempts": 3,
                        },
                        "sandbox": {"allow_level_b": False},
                    }
                    env = {
                        "XDG_CONFIG_HOME": str(
                            folder / ".control/config" / job["id"] / str(number)
                        ),
                        "UV_OFFLINE": "1",
                        **{
                            key: str(state["threads"])
                            for key in [
                                "OMP_NUM_THREADS",
                                "OPENBLAS_NUM_THREADS",
                                "MKL_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS",
                                "VECLIB_MAXIMUM_THREADS",
                            ]
                        },
                    }
                    cpus = (
                        sorted(os.sched_getaffinity(0))
                        if hasattr(os, "sched_getaffinity")
                        else []
                    )
                    affinity = (
                        cpus[
                            (len(job["attempts"]) + state["jobs"].index(job))
                            * state["threads"]
                            % len(cpus) :
                        ][: state["threads"]]
                        if cpus
                        else []
                    )
                    spec = {
                        "data": str(CODE / "benchmarks" / entry["file"]),
                        "attempt": str(attempt),
                        "exclude_columns": entry.get("evaluation_only", []),
                        "config": cfg,
                        "environment": env,
                        "affinity": affinity,
                    }
                    spec_file = folder / ".control" / f"{job['id']}-{number:03d}.json"
                    atomic_json(spec_file, spec)
                    log = (attempt.parent / f"worker-{number:03d}.log").open("w")
                    proc = subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--out",
                            str(folder),
                            "--worker",
                            str(spec_file),
                        ],
                        cwd=ROOT,
                        env={**os.environ, **env},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    log.close()
                    job.update(
                        status="running",
                        pid=proc.pid,
                        process_start=start_token(proc.pid),
                    )
                    processes[job["id"]] = proc
            write_report(folder, state)
            time.sleep(3)
        state.update(status="finished", ended_at=utc())
        write_report(folder, state)


if __name__ == "__main__":
    main()

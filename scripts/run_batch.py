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
from comparison import (
    PROVIDERS,
    read_plan,
    runtime_identity,
    select_jobs,
    validate_plan,
    validate_state,
)
from mlca.llm.usage import summary
from mlca.paths import atomic_json
from mlca.rails import load

TERMINAL = {"finished"}
# Free disk a study needs before it starts. A large table's private row-fingerprint store
# reached 4.3 GB, built through a temporary copy of the same size, plus its environment.
MIN_FREE_BYTES = 4 * 2**30
LARGE_EXTRA_BYTES = 12 * 2**30
# Never published (.gitignore) and unused once a study is scored.
PRUNED_AFTER_SCORING = (".mlca/private", ".mlca/env")


def disk_needed(job):
    return MIN_FREE_BYTES + (LARGE_EXTRA_BYTES if job["large"] else 0)


def prune(attempt):
    removed = []
    for project in sorted(p for p in attempt.iterdir() if p.is_dir()):
        for name in PRUNED_AFTER_SCORING:
            target = project / name
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
                removed.append(f"{project.name}/{name}")
    return removed


def preserve_split_manifests(attempt):
    """Retain hash-checked row membership outside ignored .mlca/data directories."""
    import csv

    from mlca.log.store import Store

    retained = []
    for project in sorted(p for p in Path(attempt).iterdir() if p.is_dir()):
        if not (project / ".mlca/log/events.jsonl").is_file():
            continue
        store = Store(project)
        store.verify()
        records = store.read()
        unlock = next(
            (i for i, r in enumerate(records) if r.type.value == "TestUnlocked"),
            len(records),
        )
        splits = [r.payload for r in records[:unlock] if r.type.value == "SplitFrozen"]
        if not splits:
            continue
        split = splits[-1]
        source = (project / split["manifest"]["path"]).resolve()
        if not source.is_relative_to(project.resolve()):
            raise ValueError("Split manifest escapes project")
        destination = Path(attempt) / f"split_manifest-{project.name}.csv"
        if not source.exists() and destination.is_file():
            source = destination
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != split["sha256"] or actual != split["manifest"]["sha256"]:
            raise ValueError("Split manifest hash mismatch")
        with source.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["row_id", "partition"]:
                raise ValueError("Unexpected split manifest columns")
            rows = list(reader)
        if len({r["row_id"] for r in rows}) != len(rows) or any(
            not r["row_id"] or r["partition"] not in {"train", "val", "test"}
            for r in rows
        ):
            raise ValueError("Invalid split membership")
        if dict(Counter(r["partition"] for r in rows)) != split["counts"]:
            raise ValueError("Split manifest counts mismatch")
        if source != destination.resolve():
            temporary = destination.with_suffix(".csv.tmp")
            shutil.copyfile(source, temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != actual:
                raise ValueError("Copied split manifest hash mismatch")
            temporary.replace(destination)
        retained.append(
            {
                "project": project.name,
                "split_id": split["split_id"],
                "path": destination.name,
                "sha256": actual,
            }
        )
    return retained


def batch_paths(parent):
    """Accept arbitrary output folder names as well as historical batch-NN folders."""
    return sorted(Path(parent).glob("*/batch.json"))


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


def validate_stage(
    folder,
    identity,
    model,
    selected,
    all_ids,
    replicates,
    rails_off,
    reasoning_effort=None,
    provider="openrouter",
    comparison_plan_sha256=None,
):
    """Require recorded successes before acceptance, and acceptance before ablation."""
    prior = []
    for path in sorted(batch_paths(folder.parent)):
        if path.parent == folder:
            continue
        state = json.loads(path.read_text())
        if (
            state.get("status") == "finished"
            and state.get("model") == model
            and state.get("provider", "openrouter") == provider
        ):
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
            and (
                comparison_plan_sha256 is None
                or state.get("comparison_plan_sha256") == comparison_plan_sha256
            )
            and len(state["jobs"]) == 3 * len(all_ids)
            and all(j.get("result", {}).get("success") for j in state["jobs"])
            for state in prior
        )
        if not accepted or set(selected) != all_ids or replicates != 1:
            raise ValueError(
                "Ablation requires successful K=3 acceptance on the same version"
            )


ROLES = {"development", "heldout", "evaluation"}


def entries(manifest=None, role="development"):
    """Tables of one role. Development and held-out tables are in MLCA's manifest;
    the biomedical evaluation has its own manifest (role evaluation)."""
    if role not in ROLES:
        raise ValueError("Unknown table role")
    path = Path(manifest) if manifest else CODE / "benchmarks/manifest.yaml"
    return [e for e in yaml.safe_load(path.read_text()) if e["role"] == role]


def data_file(entry, data_root=None):
    return Path(data_root or CODE / "benchmarks") / entry["file"]


def validate_evaluation(entries, data_root, allow_draft):
    """Evaluation tables run only from a verified data folder, with an approved task
    and a starting brief written before the run."""
    if not data_root:
        raise ValueError("Evaluation runs need --data-root (see fetch_bio_bench.py)")
    for entry in entries:
        if not allow_draft and not entry.get("approved"):
            raise ValueError("Task not approved in the preregistration: " + entry["id"])
        if entry.get("problem_type") in {None, "", "undecided"}:
            raise ValueError("Problem type undecided: " + entry["id"])
        brief = ROOT / entry["brief_file"]
        if not brief.is_file() or not brief.read_text().strip():
            raise ValueError("Starting brief missing: " + entry["brief_file"])
        for name in ["file", "external"]:
            if entry.get(name):
                path = Path(data_root) / entry[name]
                key = "sha256" if name == "file" else "external_sha256"
                if (
                    not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != entry[key]
                ):
                    raise ValueError("Data file missing or changed: " + entry[name])


def frozen(code):
    """The evaluation runs on a tagged freeze of the agent."""
    tags = subprocess.run(
        ["git", "-C", str(code), "tag", "--points-at", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [tag for tag in tags if tag.startswith("freeze")]


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
    enforced = attempt.parent / (attempt.name + "-enforced-resources.json")
    result["enforced_resources"] = (
        json.loads(enforced.read_text()) if enforced.exists() else None
    )
    if job.get("cgroup"):
        from study_resources import memory_events

        result["memory_events"] = memory_events(job["cgroup"])
        if result["memory_events"].get("oom_kill", 0):
            result.update(status="failed", category="memory_limit")
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
        splits = [r.payload for r in records if r.type.value == "SplitFrozen"]
        result["split_parameters"] = splits[-1].get("parameters") if splits else None
        result["split_sha256"] = splits[-1].get("sha256") if splits else None
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
    if attempt.is_dir():
        try:
            result["split_manifests"] = preserve_split_manifests(attempt)
        except Exception as exc:  # noqa: BLE001 -- record retention errors without rewriting measured outcomes
            # Retention failure does not rewrite the agent's measured outcome.
            result["split_manifest_error"] = {
                "error_type": type(exc).__name__,
                "message": str(exc).splitlines()[0],
            }
    result["pruned_after_scoring"] = prune(attempt) if attempt.is_dir() else []
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
    for path in batch_paths(folder.parent):
        cumulative.extend(request_records(path.parent))
    running = accounting(list({r["request_id"]: r for r in cumulative}.values()))
    state["accounting"] = costs
    state["running_accounting"] = running
    atomic_json(folder / "batch.json", state)
    lines = [
        "# "
        + state.get("stage", "development").title()
        + " batch "
        + str(state["batch"]),
        "",
        "Model, provider, runtime source and rail setting fixed for every study in this batch.",
        "Comparison plan: " + str(state.get("comparison_plan_sha256", "none")),
        "Acceptance outcome and rail effect are reported separately; failed runs are retained.",
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

    try:
        if hasattr(os, "sched_setaffinity") and spec.get("affinity"):
            os.sched_setaffinity(0, spec["affinity"])
        if spec.get("cgroup"):
            from study_resources import enter_group

            enter_group(spec["cgroup"], spec["memory_limit_mib"])
        atomic_json(
            attempt.parent / (attempt.name + "-enforced-resources.json"),
            {
                "affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
                "threads": int(spec["environment"]["OMP_NUM_THREADS"]),
                "memory_limit_mib": spec.get("memory_limit_mib"),
                "memory_enforcement": "cgroup_v2"
                if spec.get("cgroup")
                else "not_enforced",
            },
        )
        from mlca.autonomous.runner import run

        result = run(
            Path(spec["data"]),
            attempt,
            brief=spec.get("brief", ""),
            config=spec["config"],
            exclude_columns=spec["exclude_columns"],
            halt_window=0,
            external=[spec["external"]] if spec.get("external") else (),
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
    parser.add_argument("--batch", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--provider", choices=list(PROVIDERS))
    parser.add_argument("--comparison-plan", type=Path)
    parser.add_argument("--stage", choices=["acceptance", "ablation"])
    parser.add_argument(
        "--provisional-ablation",
        action="store_true",
        help="Allow a separate rails-off arm before acceptance finishes; requires a shared plan",
    )
    parser.add_argument("--shard", help="Disjoint planned study subset i/n (1-based)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-limit-mib", type=int)
    parser.add_argument(
        "--cgroup-root",
        type=Path,
        help="Existing delegated cgroup-v2 root; no automatic provisioning",
    )
    parser.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high", "max"]
    )
    parser.add_argument("--concurrency", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--deadline-s", type=float, default=5400)
    parser.add_argument("--rpm", type=float, default=60)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--tables", nargs="*")
    parser.add_argument("--replicates", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--rails-off", action="store_true")
    parser.add_argument("--role", choices=sorted(ROLES), default="development")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Task manifest (default: MLCA's benchmarks manifest)",
    )
    parser.add_argument(
        "--data-root", type=Path, help="Folder holding the manifest's files"
    )
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Smoke tests only: unapproved tasks, untagged code",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.deadline_s <= 0 or args.rpm <= 0:
        raise ValueError("Deadlines and request rates must be positive")
    if args.threads < 1 or (
        args.memory_limit_mib is not None and args.memory_limit_mib <= 0
    ):
        raise ValueError("Positive thread/memory limits required")
    if (
        (args.shard or args.provisional_ablation or args.stage)
        and not args.comparison_plan
        and not args.resume
    ):
        raise ValueError(
            "Comparison stage, sharding and concurrent ablation require --comparison-plan"
        )
    identity = source_identity()
    all_entries = entries(args.manifest, args.role)
    if args.role != "development":
        if args.comparison_plan or args.rails_off or args.stage:
            raise ValueError(
                "Held-out and evaluation runs use the frozen agent with rails on"
            )
        if not args.allow_draft and not frozen(CODE):
            raise ValueError(
                "Held-out and evaluation runs need a freeze tag on the MLCA commit"
            )
    if args.role == "evaluation":
        validate_evaluation(all_entries, args.data_root, args.allow_draft)
    plan = read_plan(args.comparison_plan) if args.comparison_plan else None
    if plan:
        validate_plan(plan, identity, all_entries, runtime_identity(CODE))
        # Configuration comes from the shared immutable plan, not per-node overrides.
        for name in ["provider", "model", "reasoning_effort"]:
            if getattr(args, name) is not None and getattr(args, name) != plan[name]:
                raise ValueError("CLI setting differs from comparison plan: " + name)
    if plan:
        for attr, key in [
            ("threads", "threads"),
            ("memory_limit_mib", "memory_limit_mib"),
            ("deadline_s", "deadline_s"),
            ("rpm", "rpm_per_node"),
        ]:
            flag = "--" + attr.replace("_", "-")
            if (
                any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])
                and getattr(args, attr) != plan[key]
            ):
                raise ValueError("CLI setting differs from comparison plan: " + attr)
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
            if state.get("comparison_plan_sha256"):
                stored = read_plan(folder / "comparison-plan.json")
                validate_plan(stored, identity, all_entries, runtime_identity(CODE))
                validate_state(stored, state)
                if state["comparison_plan_sha256"] != stored["plan_sha256"]:
                    raise ValueError("Stored comparison identity differs")
            if args.rails_off and state["rails_enabled"]:
                raise ValueError("Resume cannot change the recorded rail setting")
            if state.get("role", "development") != args.role or state.get(
                "manifest"
            ) != (str(args.manifest) if args.manifest else None):
                raise ValueError("Resume cannot change the table role or manifest")
            if plan and plan["plan_sha256"] != state.get("comparison_plan_sha256"):
                raise ValueError("Resume cannot change comparison plan")
            for name in ["provider", "model", "reasoning_effort", "stage", "shard"]:
                value = getattr(args, name)
                if value is not None and value != state.get(
                    name, "openrouter" if name == "provider" else None
                ):
                    raise ValueError("Resume cannot change recorded " + name)
            for attr in [
                "threads",
                "memory_limit_mib",
                "deadline_s",
                "rpm",
                "replicates",
            ]:
                flag = "--" + attr.replace("_", "-")
                if any(
                    a == flag or a.startswith(flag + "=") for a in sys.argv[1:]
                ) and getattr(args, attr) != state.get(attr):
                    raise ValueError("Resume cannot change recorded " + attr)
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
            if state_file.exists() or (not plan and (not args.batch or not args.model)):
                raise ValueError(
                    "New batch needs a fresh directory, --batch and --model"
                )
            if plan:
                stage = args.stage or ("ablation" if args.rails_off else "acceptance")
                if args.rails_off and stage != "ablation":
                    raise ValueError("--rails-off conflicts with acceptance stage")
                if args.provisional_ablation and stage != "ablation":
                    raise ValueError(
                        "Concurrent ablation flag applies only to ablation"
                    )
                expected_replicates = 3 if stage == "acceptance" else 1
                if (
                    any(
                        a == "--replicates" or a.startswith("--replicates=")
                        for a in sys.argv[1:]
                    )
                    and args.replicates != expected_replicates
                ):
                    raise ValueError("CLI replicate count differs from comparison plan")
                planned_jobs = select_jobs(plan, stage, args.tables, args.shard)
                selected = list(dict.fromkeys(j["dataset"] for j in planned_jobs))
                if stage == "ablation" and not args.provisional_ablation:
                    # Sequential path still requires recorded acceptance; planned path may explicitly overlap.
                    validate_stage(
                        folder,
                        identity,
                        plan["model"],
                        list(by_id),
                        set(by_id),
                        1,
                        True,
                        plan["reasoning_effort"],
                        provider=plan["provider"],
                        comparison_plan_sha256=plan["plan_sha256"],
                    )
            else:
                stage = (
                    args.role
                    if args.role != "development"
                    else "ablation"
                    if args.rails_off
                    else "acceptance"
                    if args.replicates == 3 and set(args.tables or by_id) == set(by_id)
                    else "development"  # includes pooled reruns of tables that fell short
                )
                planned_jobs = None
                selected = args.tables or list(by_id)
            if (
                not set(selected) <= set(by_id)
                or not plan
                and args.role == "development"
                and args.batch == 1
                and set(selected) != set(by_id)
                or len(selected) != len(set(selected))
            ):
                raise ValueError(
                    "Only development tables; batch-01 must include all twelve"
                )
            if not plan and args.role == "development":
                validate_stage(
                    folder,
                    identity,
                    args.model,
                    selected,
                    set(by_id),
                    args.replicates
                    if args.rails_off
                    else 3
                    if stage == "acceptance"
                    else 1,
                    args.rails_off,
                    args.reasoning_effort,
                    provider=args.provider or "openrouter",
                )
            started = folder.parent / "batch-start.json"
            if not started.exists():
                atomic_json(started, {"started_at": utc()})
            state = {
                **identity,
                "batch": args.batch or stage,
                "stage": stage,
                "role": args.role,
                "manifest": str(args.manifest) if args.manifest else None,
                "data_root": str(args.data_root) if args.data_root else None,
                "allow_draft": args.allow_draft,
                "provider": plan["provider"] if plan else args.provider or "openrouter",
                "endpoint": plan["endpoint"]
                if plan
                else PROVIDERS[args.provider or "openrouter"][0],
                "comparison_plan_sha256": plan["plan_sha256"] if plan else None,
                "shard": args.shard,
                "provisional_ablation": args.provisional_ablation,
                "datasets": plan["datasets"]
                if plan
                else [{"id": e["id"], "sha256": e["sha256"]} for e in all_entries],
                "started_at": utc(),
                "model": plan["model"] if plan else args.model,
                "reasoning_effort": plan["reasoning_effort"]
                if plan
                else args.reasoning_effort,
                "concurrency": args.concurrency,
                "deadline_s": plan["deadline_s"] if plan else args.deadline_s,
                "rpm": plan["rpm_per_node"] if plan else args.rpm,
                "rails_enabled": stage != "ablation",
                "replicates": (3 if stage == "acceptance" else 1)
                if plan
                else args.replicates,
                "threads": plan["threads"] if plan else args.threads,
                "memory_limit_mib": plan["memory_limit_mib"]
                if plan
                else args.memory_limit_mib,
                "memory_limit": "cgroup_v2"
                if (plan["memory_limit_mib"] if plan else args.memory_limit_mib)
                else "not_enforced",
                "runtime": plan["runtime"] if plan else runtime_identity(CODE),
                "sandbox": "A",
                "spending_ceiling_usd": None,
                "model_call_deadline_s": 300,
                "temperature": 0,
                "max_output_tokens": plan["max_output_tokens"]
                if plan
                else args.max_output_tokens,
                "structured_outputs": True,
                "sdk_timeout_s": 90,
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
                        **j,
                        "large": by_id[j["dataset"]]["rows"] > 500000,
                        "status": "queued",
                        "attempts": [],
                    }
                    for j in (
                        planned_jobs
                        or [
                            {
                                "id": f"{name}-{rep:03d}",
                                "dataset": name,
                                "replicate": rep,
                            }
                            for name in selected
                            for rep in range(1, args.replicates + 1)
                        ]
                    )
                ],
            }
            if plan:
                atomic_json(folder / "comparison-plan.json", plan)
        if plan:
            validate_state(plan, state)
        provider_name = state.get("provider", "openrouter")
        state.setdefault("endpoint", PROVIDERS[provider_name][0])
        state.setdefault("provider", provider_name)
        memory_mib = state.get("memory_limit_mib")
        if memory_mib and not args.cgroup_root:
            raise ValueError(
                "Memory enforcement requires --cgroup-root; refusing an unenforced ceiling"
            )
        if (
            hasattr(os, "sched_getaffinity")
            and len(os.sched_getaffinity(0)) < state["threads"]
        ):
            raise ValueError(
                "Insufficient available CPUs for requested per-study affinity"
            )
        if Path("/proc/meminfo").is_file():
            state["machine"]["memory_total_kib"] = int(
                Path("/proc/meminfo").read_text().split("MemTotal:")[1].split()[0]
            )
        from mlca.sandbox.level import detect

        if detect(False) != "A":
            raise ValueError("Sandbox A required")
        if state.get("comparison_plan_sha256"):
            state["rate_limit_dir"] = str(
                Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
                / "mlca-comparison-rate"
                / state["comparison_plan_sha256"]
            )
        else:
            state["rate_limit_dir"] = str(folder / ".control/rate")
        key_env = PROVIDERS[provider_name][1]
        if not os.environ.get(key_env):
            raise ValueError("Provide " + key_env + " in the environment")
        if memory_mib:
            from study_resources import prepare_group

            probe = prepare_group(
                args.cgroup_root, "mlca-probe-" + str(os.getpid()), memory_mib
            )
            probe.rmdir()
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
                state.get("stage", "development") == "development"
                and (
                    datetime.now(timezone.utc) - datetime.fromisoformat(started)
                ).total_seconds()
                > 5 * 86400
            ):
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
                launched = []
                for job in choose_next(state["jobs"], active, state["concurrency"]):
                    if (
                        job.get("first_started")
                        and time.time() - job["first_started"] > state["deadline_s"]
                    ):
                        job["stop_reason"] = "study_deadline"
                        finish_job(folder, job, by_id[job["dataset"]])
                        continue
                    if shutil.disk_usage(folder).free < disk_needed(job):
                        # Wait for running studies to finish and be pruned.
                        continue
                    entry = by_id[job["dataset"]]
                    if (
                        hashlib.sha256(
                            data_file(entry, state.get("data_root")).read_bytes()
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
                                "kind": provider_name,
                                "base_url": state["endpoint"],
                                "api_key_env": key_env,
                                "timeout_s": state.get("sdk_timeout_s", 90),
                                "structured_outputs": state["structured_outputs"],
                                "call_deadline_s": state["model_call_deadline_s"],
                                "requests_per_minute": state["rpm"],
                                "rate_limit_dir": state["rate_limit_dir"],
                            }
                        },
                        "roles": {
                            role: {
                                "provider": "development",
                                "model": state["model"],
                                "temperature": state.get("temperature", 0),
                                "max_output_tokens": state.get(
                                    "max_output_tokens", 16384
                                ),
                                **(
                                    {"reasoning_effort": state["reasoning_effort"]}
                                    if state.get("reasoning_effort")
                                    else {}
                                ),
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
                    offset = state["jobs"].index(job) * state["threads"]
                    affinity = (
                        [
                            cpus[(offset + n) % len(cpus)]
                            for n in range(state["threads"])
                        ]
                        if cpus
                        else []
                    )
                    cgroup = None
                    if memory_mib:
                        from study_resources import prepare_group

                        group_name = (
                            "mlca-"
                            + hashlib.sha256(
                                (str(folder) + job["id"] + str(number)).encode()
                            ).hexdigest()[:24]
                        )
                        try:
                            cgroup = prepare_group(
                                args.cgroup_root, group_name, memory_mib
                            )
                        except (OSError, ValueError) as exc:
                            # Retain the failed study without abandoning other active workers.
                            atomic_json(
                                attempt / "worker-result.json",
                                {
                                    "status": "failed",
                                    "category": "resource_setup_error",
                                    "error_type": type(exc).__name__,
                                },
                            )
                            job["cgroup"] = None
                            finish_job(folder, job, entry)
                            continue
                    state["memory_limit"] = "cgroup_v2" if cgroup else "not_enforced"
                    spec = {
                        "data": str(data_file(entry, state.get("data_root"))),
                        "brief": (ROOT / entry["brief_file"]).read_text().strip()
                        if entry.get("brief_file")
                        else "",
                        "external": str(Path(state["data_root"]) / entry["external"])
                        if entry.get("external")
                        else None,
                        "attempt": str(attempt),
                        "exclude_columns": entry.get("evaluation_only", []),
                        "config": cfg,
                        "environment": env,
                        "affinity": affinity,
                        "memory_limit_mib": memory_mib,
                        "cgroup": str(cgroup) if cgroup else None,
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
                        cgroup=str(cgroup) if cgroup else None,
                    )
                    processes[job["id"]] = proc
                    launched.append(job["id"])
                if (
                    not active
                    and not launched
                    and any(j["status"] == "queued" for j in state["jobs"])
                ):
                    # Nothing running can free space: stop clearly instead of waiting forever.
                    state.update(status="stopped", stop_reason="insufficient_disk")
                    write_report(folder, state)
                    print(json.dumps({"stopped": "insufficient_disk"}), flush=True)
                    return
            write_report(folder, state)
            time.sleep(3)
        state.update(status="finished", ended_at=utc())
        write_report(folder, state)


if __name__ == "__main__":
    main()

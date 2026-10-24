"""Resumable process queue for author-approved development batches; never analysis code."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
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
    provider_kind,
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


def git_tree(root, path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", f"HEAD:{path}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def get_host_info():
    return {
        "hostname": socket.gethostname(),
        "cpus": os.cpu_count() or 1,
    }


def probe_endpoint(endpoint_url, timeout=5.0):
    try:
        parsed = urllib.parse.urlparse(endpoint_url)
        host = parsed.hostname
        if not host:
            return True
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except OSError:
        return False


def run_env_preflight():
    from env_preflight import preflight

    return {**preflight(), "at": utc(), "host": get_host_info()}


def source_identity():
    # Only the code that runs counts. Run folders are tracked evidence and change during a resume.
    for root, paths in [(CODE, ["src", "templates", "rails"]), (ROOT, ["scripts"])]:
        if subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", *paths], check=False
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
        "controller_tree": git_tree(ROOT, "scripts"),
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


def provider_errors(records):
    """Payload-free transport diagnostics, including unknown status codes as null."""
    fields = (
        "request_id",
        "error_type",
        "category",
        "status_code",
        "latency_ms",
        "pacing_wait_s",
        "rate_limit_waits",
    )
    return [
        {name: record.get(name) for name in fields}
        for record in records
        if record.get("status") == "error"
    ]


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
    result["provider_errors"] = provider_errors(list(request_list.values()))
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
    spec_file = folder / ".control" / f"{job['id']}-{len(job['attempts']):03d}.json"
    launch_host = get_host_info()
    launch_commit = None
    if spec_file.exists():
        try:
            stored_spec = json.loads(spec_file.read_text())
            launch_host = stored_spec.get("host") or launch_host
            launch_commit = stored_spec.get("controller_commit")
        except (OSError, json.JSONDecodeError):
            pass
    if not launch_commit:
        state_file = folder / "batch.json"
        if state_file.exists():
            try:
                launch_commit = json.loads(state_file.read_text()).get("controller_commit")
            except (OSError, json.JSONDecodeError):
                pass
    if not launch_commit:
        launch_commit = git_head(ROOT)
    prov = {
        "host": launch_host,
        "controller_commit": launch_commit,
        "attempt": job["attempts"][-1],
    }
    result["host"] = launch_host
    result["controller_commit"] = launch_commit
    result["provenance"] = prov
    if attempt.is_dir():
        prov_file = attempt / "provenance.json"
        if not prov_file.exists():
            atomic_json(prov_file, prov)
    job.update(
        status="interrupted"
        if job.get("stop_reason") in INTERRUPT_REASONS
        else "finished",
        finished_at=utc(),
        result=result,
    )


def is_connection_failure(job):
    res = job.get("result") or {}
    if res.get("category") != "provider_error":
        return False
    if res.get("furthest_event") != "setup":
        return False
    u = res.get("usage", {})
    known = u.get("known", {})
    if (known.get("input_tokens") or 0) > 0 or (known.get("output_tokens") or 0) > 0:
        return False
    msg = str(res.get("message") or "")
    err_type = str(res.get("error_type") or "")
    conn_indicators = [
        "Connection error",
        "ConnectionError",
        "APIConnectionError",
        "connection error",
    ]
    if any(ind in msg or ind in err_type for ind in conn_indicators):
        return True
    p_errs = res.get("provider_errors") or []
    for pe in p_errs:
        if any(
            ind in str(pe.get("error_type", "")) or ind in str(pe.get("message", ""))
            for ind in conn_indicators
        ):
            return True
    return False


INTERRUPT_REASONS = {"interrupted_by_author", "machine_shutdown", "source_changed_during_batch"}
ENVIRONMENT_FAILURE = ["Project environment command failed", "uv is required"]


def is_environment_failure(job):
    res = job.get("result") or {}
    return res.get("furthest_event") == "setup" and any(
        text in str(res.get("message") or "") for text in ENVIRONMENT_FAILURE
    )


SANDBOX_SETUP_FAILURE = [
    "bwrap: loopback",
    "bwrap: setting up uid map",
    "bwrap: No permissions to create new namespace",
    "bwrap: Creating new namespace failed",
]


def is_sandbox_setup_failure(job, folder):
    """The sandbox could not start on this machine (e.g. Ubuntu's user-namespace restriction).

    Evidence is bubblewrap's own setup error in the attempt's step stderr, not the analysis code.
    """
    res = job.get("result") or {}
    if folder is None or res.get("category") != "sandbox_denied" or not job.get("attempts"):
        return False
    attempt = Path(folder) / job["attempts"][-1]
    for stderr in attempt.glob("run-*/.mlca/steps/*/attempts/*/stderr.txt"):
        text = stderr.read_text(errors="replace")
        if any(marker in text for marker in SANDBOX_SETUP_FAILURE):
            return True
    return False


def infrastructure_reason(job, folder=None):
    """Failures caused by the machine or network, not by the agent's analysis; None otherwise."""
    if is_connection_failure(job):
        return "connection_failure"
    if is_environment_failure(job):
        return "environment_failure"
    if is_sandbox_setup_failure(job, folder):
        return "sandbox_setup_failure"
    return None


def needs_operator(job, folder=None):
    """Finished jobs that pause the scheduler: infrastructure or any provider error."""
    res = job.get("result") or {}
    return bool(infrastructure_reason(job, folder)) or res.get("category") == "provider_error"


def requeue(job, reason):
    """Queue a new attempt; the old attempt stays. The study deadline restarts for it."""
    job.setdefault("retries", []).append(
        {
            "retry_of": job["attempts"][-1] if job.get("attempts") else None,
            "reason": reason,
            "at": utc(),
            "prior_result": job.get("result"),
        }
    )
    if "first_started" in job:
        job.setdefault("first_started_original", job["first_started"])
        job.pop("first_started")
    job["status"] = "queued"
    for key in ["result", "stop_reason", "stopping_at", "pid", "process_start", "cgroup"]:
        job.pop(key, None)


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
        for retry in job.get("retries", []):
            prior = retry.get("prior_result") or {}
            pu = prior.get("usage", {}).get("known", {})
            pcriteria = prior.get("leakage", {}).get("criteria", {})
            pflags = " / ".join(
                str(pcriteria.get(k, {}).get("value"))
                for k in ["fitted_columns", "group_disjoint", "time_ordered"]
            )
            attempt_label = retry.get("retry_of", "attempt-001")
            lines.append(
                f"| {job['dataset']} / {job['replicate']} ({attempt_label}) | {prior.get('workflow_completed', False)} | {prior.get('setup_agreement')} | {pflags} | {prior.get('success', False)} | {prior.get('furthest_event', 'setup')} / {prior.get('furthest_step')} | {prior.get('category', '')} | {prior.get('repair_calls', 0)} | {prior.get('usage', {}).get('requests', 0)} | {pu.get('input_tokens')} / {pu.get('output_tokens')} / {pu.get('reasoning_tokens')} | {pu.get('cost_usd')} | {prior.get('wall_s')} | {json.dumps(prior.get('usage', {}).get('models', {}))} |"
            )
        r = job.get("result", {})
        u = r.get("usage", {}).get("known", {})
        criteria = r.get("leakage", {}).get("criteria", {})
        flags = " / ".join(
            str(criteria.get(k, {}).get("value"))
            for k in ["fitted_columns", "group_disjoint", "time_ordered"]
        )
        attempt_suffix = (
            f" (attempt-{len(job.get('attempts', [1])):03d})"
            if job.get("retries")
            else ""
        )
        lines.append(
            f"| {job['dataset']} / {job['replicate']}{attempt_suffix} | {r.get('workflow_completed', job['status'])} | {r.get('setup_agreement')} | {flags} | {r.get('success')} | {r.get('furthest_event', 'setup')} / {r.get('furthest_step')} | {r.get('category', '')} | {r.get('repair_calls')} | {r.get('usage', {}).get('requests')} | {u.get('input_tokens')} / {u.get('output_tokens')} / {u.get('reasoning_tokens')} | {u.get('cost_usd')} | {r.get('wall_s')} | {json.dumps(r.get('usage', {}).get('models', {}))} |"
        )
    retried_jobs = [j for j in state["jobs"] if j.get("retries")]
    if retried_jobs:
        lines += [
            "",
            "## Retried attempts",
            "",
            "These attempts ended for infrastructure reasons (connection or environment failure at setup, machine shutdown or loss, operator interrupt) and were run again as new attempts; the earlier attempts are kept and listed above:",
            "",
            "| Job | Dataset | Replicate | Retried attempt | Reason | Retried at |",
            "|---|---|---:|---|---|---|",
        ]
        for j in retried_jobs:
            for ret in j["retries"]:
                lines.append(
                    f"| {j['id']} | {j['dataset']} | {j['replicate']} | {ret.get('retry_of')} | {ret.get('reason')} | {ret.get('at')} |"
                )
    if state.get("operator_pauses"):
        lines += [
            "",
            "## Scheduler paused for the operator",
            "",
            "| Job | Reason | Message | At |",
            "|---|---|---|---|",
        ] + [
            f"| {o['job']} | {o['reason']} | {o['message'][:200].replace('|', '/')} | {o['at']} |"
            for o in state["operator_pauses"]
        ]
    if state.get("env_preflights"):
        last = state["env_preflights"][-1]
        lines += [
            "",
            "## Environment preflight",
            "",
            f"Last check {last['at']} on {last['host']['hostname']}, Python {last['python']}: "
            + ", ".join(
                f"{k} {v['lock_sha256'][:12]}" + (" (matches reference)" if v["matches_reference"] else "")
                for k, v in last["sets"].items()
            ),
        ]
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
    parser.add_argument(
        "--concurrency",
        type=int,
        choices=range(1, 7),
        help="Concurrent studies (Ollama: 1; other providers: 6)",
    )
    parser.add_argument("--deadline-s", type=float, default=5400)
    parser.add_argument("--rpm", type=float, default=60)
    parser.add_argument(
        "--sdk-timeout-s",
        type=float,
        help="SDK network timeout (Ollama: 300s; other providers: 90s)",
    )
    parser.add_argument(
        "--model-call-deadline-s",
        type=float,
        help="Hard provider-call deadline (Ollama: 600s; other providers: 300s)",
    )
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--structured-outputs", choices=["on", "off"], default="on")
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
    parser.add_argument(
        "--controller-upgrade",
        help="Reason for upgrading batch controller commit on resume",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Gracefully pause batch by preventing new study launches",
    )
    parser.add_argument(
        "--list-connection-failures",
        action="store_true",
        help="List jobs eligible for one-time connection failure retry and exit",
    )
    parser.add_argument(
        "--retry-connection-failures",
        action="store_true",
        help="Re-queue eligible connection failures once (requires --resume)",
    )
    parser.add_argument(
        "--list-infrastructure-failures",
        action="store_true",
        help="List jobs whose last attempt failed for connection or environment reasons and exit",
    )
    parser.add_argument(
        "--retry-infrastructure-failures",
        action="store_true",
        help="Re-queue connection, environment and sandbox-setup failures, once per reason (requires --resume)",
    )
    parser.add_argument(
        "--skip-env-preflight",
        action="store_true",
        help="Do not build the offline study environments before launching (not for evaluation)",
    )
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.pause:
        folder = args.out.resolve()
        ctrl = folder / ".control"
        ctrl.mkdir(parents=True, exist_ok=True)
        (ctrl / "PAUSE").touch()
        print(json.dumps({"pause_requested": str(folder)}), flush=True)
        return
    if args.list_connection_failures:
        folder = args.out.resolve()
        state_file = folder / "batch.json"
        if not state_file.exists():
            raise FileNotFoundError(f"No batch.json found in {folder}")
        state = json.loads(state_file.read_text())
        matching = [j for j in state.get("jobs", []) if is_connection_failure(j)]
        for j in matching:
            err = (j.get("result") or {}).get("message") or ""
            print(f"{j['id']} ({j['dataset']} replicate {j['replicate']}): {err}")
        return
    if args.list_infrastructure_failures:
        state_file = args.out.resolve() / "batch.json"
        if not state_file.exists():
            raise FileNotFoundError(f"No batch.json found in {args.out.resolve()}")
        for j in json.loads(state_file.read_text()).get("jobs", []):
            if reason := infrastructure_reason(j, args.out.resolve()):
                err = (j.get("result") or {}).get("message") or ""
                print(f"{j['id']} ({j['dataset']} replicate {j['replicate']}) {reason}: {err[:200]}")
        return
    if args.retry_infrastructure_failures and not args.resume:
        raise ValueError("--retry-infrastructure-failures requires --resume")
    if args.retry_connection_failures and not args.resume:
        raise ValueError("--retry-connection-failures requires --resume")
    if args.controller_upgrade and not args.resume:
        raise ValueError("--controller-upgrade requires --resume")
    # New-batch defaults only. Resume still uses its recorded state; plans win below.
    ollama = args.provider == "ollama"
    if args.sdk_timeout_s is None:
        args.sdk_timeout_s = 300 if ollama else 90
    if args.model_call_deadline_s is None:
        args.model_call_deadline_s = 600 if ollama else 300
    if args.concurrency is None:
        args.concurrency = 1 if ollama else 6
    if any(
        not (0 < value < float("inf"))
        for value in [
            args.deadline_s,
            args.rpm,
            args.sdk_timeout_s,
            args.model_call_deadline_s,
        ]
    ):
        raise ValueError(
            "Deadlines, timeouts and request rates must be positive and finite"
        )
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
    if args.skip_env_preflight and args.role == "evaluation":
        raise ValueError("Evaluation batches always run the environment preflight")
    env_check = None if args.skip_env_preflight else run_env_preflight()
    plan = read_plan(args.comparison_plan) if args.comparison_plan else None
    if plan:
        validate_identity = (
            {k: identity[k] for k in plan["identity"]}
            if set(plan["identity"]).issubset(identity)
            else identity
        )
        validate_plan(plan, validate_identity, all_entries, runtime_identity(CODE))
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
            ("sdk_timeout_s", "sdk_timeout_s"),
            ("model_call_deadline_s", "model_call_deadline_s"),
        ]:
            flag = "--" + attr.replace("_", "-")
            if (
                any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])
                and getattr(args, attr) != plan[key]
            ):
                raise ValueError("CLI setting differs from comparison plan: " + attr)
    structured_explicit = any(
        a == "--structured-outputs" or a.startswith("--structured-outputs=")
        for a in sys.argv[1:]
    )
    if (
        plan
        and structured_explicit
        and (args.structured_outputs == "on") != plan["structured_outputs"]
    ):
        raise ValueError("CLI setting differs from comparison plan: structured_outputs")
    by_id = {e["id"]: e for e in all_entries}
    folder = args.out.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ".control").mkdir(exist_ok=True)
    interrupted = [False]
    shutdown = [False]

    def _on_interrupt(*_):
        interrupted[0] = True

    def _on_shutdown(*_):
        # VM stop or Spot preemption: running attempts are re-queued on resume.
        shutdown[0] = True
        interrupted[0] = True

    signal.signal(signal.SIGINT, _on_interrupt)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_shutdown)
    with FileLock(str(folder / ".control/controller.lock"), timeout=0):
        state_file = folder / "batch.json"
        if args.resume:
            state = json.loads(state_file.read_text())
            if state["code_commit"] != identity["code_commit"]:
                raise ValueError("Resume requires the original code commit")
            if state["rails"] != identity["rails"]:
                raise ValueError("Resume requires the original rails version and hash")
            curr_ctl = identity["controller_commit"]
            stored_ctl = state.get("controller_commit")
            if curr_ctl != stored_ctl:
                curr_tree = identity.get("controller_tree")
                stored_tree = state.get("controller_tree")
                tree_unchanged = (
                    stored_tree is not None
                    and curr_tree is not None
                    and curr_tree == stored_tree
                )
                if tree_unchanged:
                    state.setdefault("controller_history", []).append({
                        "from": stored_ctl,
                        "to": curr_ctl,
                        "at": utc(),
                        "reason": "outside_scripts_commit",
                        "host": get_host_info(),
                    })
                    state["controller_commit"] = curr_ctl
                    state["controller_tree"] = curr_tree
                elif args.controller_upgrade:
                    state.setdefault("controller_history", []).append({
                        "from": stored_ctl,
                        "to": curr_ctl,
                        "at": utc(),
                        "reason": args.controller_upgrade,
                        "host": get_host_info(),
                    })
                    state["controller_commit"] = curr_ctl
                    state["controller_tree"] = curr_tree
                else:
                    raise ValueError(
                        "Resume requires the original code, rails and controller version"
                    )
            else:
                if "controller_tree" not in state and identity.get("controller_tree"):
                    state["controller_tree"] = identity["controller_tree"]
            if state.get("stop_reason"):
                # The previous stop is kept in its jobs' attempts; this resume starts without it.
                state.setdefault("stop_history", []).append(
                    {"stop_reason": state.pop("stop_reason"), "resumed_at": utc()}
                )
            pause_file = folder / ".control/PAUSE"
            if pause_file.exists():
                pause_file.unlink()
            if state.get("status") == "paused":
                state["status"] = "running"
            pauses = state.setdefault("pauses", [])
            if pauses and "resumed_at" not in pauses[-1]:
                pauses[-1]["resumed_at"] = utc()
                pauses[-1]["resumed_by_host"] = get_host_info()
            if not state.get("comparison_plan_sha256"):
                state["rate_limit_dir"] = str(folder / ".control/rate")
            if args.retry_connection_failures:
                for job in state["jobs"]:
                    if is_connection_failure(job):
                        if any(
                            r.get("reason") == "connection_failure"
                            for r in job.get("retries", [])
                        ):
                            raise ValueError(
                                f"Job {job['id']} already retried once; refusing second retry"
                            )
                        last_attempt = (
                            job["attempts"][-1]
                            if job.get("attempts")
                            else "attempt-001"
                        )
                        retry_info = {
                            "retry_of": last_attempt,
                            "reason": "connection_failure",
                            "at": utc(),
                            "prior_result": job.get("result"),
                        }
                        job.setdefault("retries", []).append(retry_info)
                        if "first_started" in job:
                            job["first_started_original"] = job["first_started"]
                            job.pop("first_started", None)
                        job["status"] = "queued"
                        job.pop("result", None)
                        job.pop("stop_reason", None)
                        job.pop("stopping_at", None)
                        job.pop("pid", None)
                        job.pop("process_start", None)
                        job.pop("cgroup", None)
            if state.get("comparison_plan_sha256"):
                stored = read_plan(folder / "comparison-plan.json")
                validate_identity = (
                    {k: identity[k] for k in stored["identity"]}
                    if set(stored["identity"]).issubset(identity)
                    else identity
                )
                validate_plan(
                    stored, validate_identity, all_entries, runtime_identity(CODE)
                )
                validate_state(stored, state)
                if state["comparison_plan_sha256"] != stored["plan_sha256"]:
                    raise ValueError("Stored comparison identity differs")
            if (
                structured_explicit
                and (args.structured_outputs == "on") != state["structured_outputs"]
            ):
                raise ValueError("Resume cannot change recorded structured_outputs")
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
                "sdk_timeout_s",
                "model_call_deadline_s",
            ]:
                flag = "--" + attr.replace("_", "-")
                if any(
                    a == flag or a.startswith(flag + "=") for a in sys.argv[1:]
                ) and getattr(args, attr) != state.get(attr):
                    raise ValueError("Resume cannot change recorded " + attr)
            if args.retry_infrastructure_failures:
                for job in state["jobs"]:
                    reason = infrastructure_reason(job, folder)
                    if not reason:
                        continue
                    if any(r.get("reason") == reason for r in job.get("retries", [])):
                        raise ValueError(
                            f"Job {job['id']} already retried once for {reason}; refusing second retry"
                        )
                    requeue(job, reason)
            for job in state["jobs"]:
                if job["status"] == "interrupted":
                    requeue(job, job.get("stop_reason") or "interrupted")
                if job["status"] in {"running", "stopping"} and not alive(job):
                    attempt = folder / job["attempts"][-1]
                    if (attempt / "worker-result.json").exists():
                        finish_job(folder, job, by_id[job["dataset"]])
                        if job["status"] == "interrupted":
                            requeue(job, job.get("stop_reason") or "interrupted")
                    else:
                        # Controller and workers vanished together (VM lost or killed).
                        requeue(job, "machine_lost")
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
                if args.role == "development" and not args.rails_off and frozen(CODE):
                    stage = "replication"
                planned_jobs = None
                selected = args.tables or list(by_id)
            if (
                not set(selected) <= set(by_id)
                or not plan
                and stage == "development"
                and args.batch == 1
                and set(selected) != set(by_id)
                or len(selected) != len(set(selected))
            ):
                raise ValueError(
                    "Only development tables; batch-01 must include every development table"
                )
            if not plan and args.role == "development" and stage != "replication":
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
                "model_call_deadline_s": plan["model_call_deadline_s"]
                if plan
                else args.model_call_deadline_s,
                "temperature": 0,
                "max_output_tokens": plan["max_output_tokens"]
                if plan
                else args.max_output_tokens,
                "structured_outputs": plan["structured_outputs"]
                if plan
                else args.structured_outputs == "on",
                "sdk_timeout_s": plan["sdk_timeout_s"] if plan else args.sdk_timeout_s,
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
        if env_check:
            state.setdefault("env_preflights", []).append(env_check)
        processes = {}
        network_outage_start = None
        while any(j["status"] not in TERMINAL for j in state["jobs"]):
            try:
                current = source_identity()
                # Commits of evidence or documents by other agents move HEAD but not the code that runs.
                unchanged = all(
                    current.get(k) == identity.get(k)
                    for k in ["code_commit", "controller_tree", "rails"]
                )
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
                    if job["status"] == "finished" and needs_operator(job, folder):
                        (folder / ".control/PAUSE").touch()
                        state.setdefault("operator_pauses", []).append(
                            {
                                "job": job["id"],
                                "reason": infrastructure_reason(job, folder)
                                or job["result"].get("category"),
                                "message": str(job["result"].get("message") or "")[:500],
                                "at": utc(),
                                "host": get_host_info(),
                            }
                        )
                        print(
                            json.dumps(
                                {
                                    "OPERATOR_NEEDED": job["id"],
                                    "reason": state["operator_pauses"][-1]["reason"],
                                    "action": "scheduler paused; running studies finish",
                                }
                            ),
                            flush=True,
                        )
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
                        "study_deadline"
                        if expired
                        else "machine_shutdown"
                        if shutdown[0]
                        else "source_changed_during_batch"
                        if state.get("stop_reason") == "source_changed_during_batch"
                        else "interrupted_by_author"
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
            pause_file = folder / ".control/PAUSE"
            pause_requested = pause_file.exists()
            if interrupted[0] and not active:
                state["status"] = "interrupted"
                if network_outage_start is not None:
                    outage_end = time.time()
                    state.setdefault("network_waits", []).append({
                        "start": datetime.fromtimestamp(
                            network_outage_start, timezone.utc
                        ).isoformat(),
                        "end": datetime.fromtimestamp(
                            outage_end, timezone.utc
                        ).isoformat(),
                        "seconds": round(outage_end - network_outage_start, 2),
                        "endpoint": state["endpoint"],
                    })
                write_report(folder, state)
                return
            if pause_requested and not active and not interrupted[0]:
                state["status"] = "paused"
                state.setdefault("pauses", []).append({
                    "paused_at": utc(),
                    "host": get_host_info(),
                })
                if network_outage_start is not None:
                    outage_end = time.time()
                    state.setdefault("network_waits", []).append({
                        "start": datetime.fromtimestamp(
                            network_outage_start, timezone.utc
                        ).isoformat(),
                        "end": datetime.fromtimestamp(
                            outage_end, timezone.utc
                        ).isoformat(),
                        "seconds": round(outage_end - network_outage_start, 2),
                        "endpoint": state["endpoint"],
                    })
                write_report(folder, state)
                print(json.dumps({"status": "paused"}), flush=True)
                return
            if not interrupted[0] and not pause_requested:
                candidates = choose_next(state["jobs"], active, state["concurrency"])
                if candidates:
                    if not probe_endpoint(state["endpoint"]):
                        if network_outage_start is None:
                            network_outage_start = time.time()
                            print(
                                json.dumps({
                                    "network_status": "endpoint_unreachable",
                                    "endpoint": state["endpoint"],
                                }),
                                flush=True,
                            )
                        for _ in range(30):
                            if interrupted[0] or (folder / ".control/PAUSE").exists():
                                break
                            time.sleep(1)
                        continue
                    elif network_outage_start is not None:
                        outage_end = time.time()
                        wait_entry = {
                            "start": datetime.fromtimestamp(
                                network_outage_start, timezone.utc
                            ).isoformat(),
                            "end": datetime.fromtimestamp(
                                outage_end, timezone.utc
                            ).isoformat(),
                            "seconds": round(outage_end - network_outage_start, 2),
                            "endpoint": state["endpoint"],
                        }
                        state.setdefault("network_waits", []).append(wait_entry)
                        print(
                            json.dumps({
                                "network_status": "restored",
                                "waited_s": wait_entry["seconds"],
                            }),
                            flush=True,
                        )
                        network_outage_start = None

                    launched = []
                    for job in candidates:
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
                                    "kind": provider_kind(provider_name),
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
                            "host": get_host_info(),
                            "controller_commit": state["controller_commit"],
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
        if network_outage_start is not None:
            outage_end = time.time()
            state.setdefault("network_waits", []).append({
                "start": datetime.fromtimestamp(
                    network_outage_start, timezone.utc
                ).isoformat(),
                "end": datetime.fromtimestamp(outage_end, timezone.utc).isoformat(),
                "seconds": round(outage_end - network_outage_start, 2),
                "endpoint": state["endpoint"],
            })
        state.update(status="finished", ended_at=utc())
        write_report(folder, state)


if __name__ == "__main__":
    main()

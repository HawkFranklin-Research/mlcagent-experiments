"""Immutable research-side plans for separate rails-on/off studies. No inference."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from mlca.paths import canonical

PROVIDERS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
}


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def runtime_identity(code):
    packages = sorted(
        {
            (d.metadata["Name"].lower(), d.version)
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        }
    )
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "architecture": platform.machine(),
        "packages_sha256": digest(packages),
        "lock_sha256": hashlib.sha256((code / "uv.lock").read_bytes()).hexdigest(),
    }


def build_plan(
    identity,
    datasets,
    *,
    provider,
    model,
    effort,
    threads=2,
    memory_limit_mib=None,
    runtime=None,
    deadline_s=5400,
    rpm=60,
):
    if provider not in PROVIDERS or not model or threads < 1:
        raise ValueError("Valid provider, model and positive thread count required")
    if memory_limit_mib is not None and memory_limit_mib <= 0:
        raise ValueError("Memory limit must be positive or unset")
    if deadline_s <= 0 or rpm <= 0:
        raise ValueError("Positive deadline and pacing required")
    names = [d["id"] for d in datasets]
    if len(names) != len(set(names)) or not names:
        raise ValueError("Unique nonempty development dataset list required")
    jobs = []
    for stage, count in [("acceptance", 3), ("ablation", 1)]:
        for name in names:
            for rep in range(1, count + 1):
                local = f"{name}-{rep:03d}"
                jobs.append(
                    {
                        "study_id": f"{stage}/{local}",
                        "id": local,
                        "dataset": name,
                        "replicate": rep,
                        "stage": stage,
                        "rails_enabled": stage == "acceptance",
                        "protocol_seed": 42,
                        "paired_study_id": f"acceptance/{name}-001"
                        if stage == "ablation"
                        else None,
                    }
                )
    body = {
        "schema_version": "1",
        "identity": identity,
        "runtime": runtime,
        "provider": provider,
        "endpoint": PROVIDERS[provider][0],
        "model": model,
        "reasoning_effort": effort,
        "temperature": 0,
        "max_output_tokens": 16384,
        "structured_outputs": True,
        "sdk_timeout_s": 90,
        "sandbox": "A",
        "threads": threads,
        "memory_limit_mib": memory_limit_mib,
        "memory_enforcement": "cgroup_v2" if memory_limit_mib else "not_enforced",
        "deadline_s": deadline_s,
        "model_call_deadline_s": 300,
        "script_timeout_s": 300,
        "max_worker_attempts": 3,
        "rpm_per_node": rpm,
        "spending_ceiling_usd": None,
        "concurrent_arms_allowed": True,
        "seed_policy": "Fresh agent runs, fixed protocol seed 42; actual split seeds recorded separately",
        "datasets": [
            {
                "id": d["id"],
                "sha256": d["sha256"],
                "evaluation_only": d.get("evaluation_only", []),
            }
            for d in datasets
        ],
        "jobs": jobs,
    }
    return {**body, "plan_sha256": digest(body)}


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if plan.get("plan_sha256") != digest(body):
        raise ValueError("Comparison plan hash mismatch")
    expected = build_plan(
        plan["identity"],
        plan["datasets"],
        provider=plan["provider"],
        model=plan["model"],
        effort=plan["reasoning_effort"],
        threads=plan["threads"],
        memory_limit_mib=plan["memory_limit_mib"],
        runtime=plan["runtime"],
        deadline_s=plan["deadline_s"],
        rpm=plan["rpm_per_node"],
    )
    if any(plan.get(k) != v for k, v in expected.items() if k != "plan_sha256"):
        raise ValueError(
            "Comparison plan has inconsistent settings or study assignments"
        )
    return plan


def select_jobs(plan, stage, tables=None, shard=None):
    if stage not in {"acceptance", "ablation"}:
        raise ValueError("Comparison stage must be acceptance or ablation")
    known = {d["id"] for d in plan["datasets"]}
    if tables and (len(tables) != len(set(tables)) or not set(tables) <= known):
        raise ValueError("Only unique planned development tables are allowed")
    jobs = [dict(j) for j in plan["jobs"] if j["stage"] == stage]
    # Shards partition the full arm before optional table filtering.
    if shard:
        try:
            index, total = map(int, shard.split("/"))
        except (ValueError, AttributeError):
            raise ValueError("Shard must be i/n, with 1 <= i <= n") from None
        if not 1 <= index <= total:
            raise ValueError("Shard must be i/n, with 1 <= i <= n")
        jobs = [j for n, j in enumerate(jobs) if n % total == index - 1]
    if tables:
        jobs = [j for j in jobs if j["dataset"] in tables]
    if not jobs:
        raise ValueError("Selection contains no planned studies")
    return jobs


def validate_plan(plan, identity, datasets, runtime):
    if plan["identity"] != identity or plan["runtime"] != runtime:
        raise ValueError(
            "Comparison requires identical code, controller, rails and runtime"
        )
    expected = [
        {
            "id": d["id"],
            "sha256": d["sha256"],
            "evaluation_only": d.get("evaluation_only", []),
        }
        for d in datasets
    ]
    if plan["datasets"] != expected:
        raise ValueError("Development manifest differs from comparison plan")


BATCH_SETTINGS = {
    "provider": "provider",
    "temperature": "temperature",
    "max_output_tokens": "max_output_tokens",
    "endpoint": "endpoint",
    "model": "model",
    "reasoning_effort": "reasoning_effort",
    "threads": "threads",
    "memory_limit_mib": "memory_limit_mib",
    "runtime": "runtime",
    "deadline_s": "deadline_s",
    "model_call_deadline_s": "model_call_deadline_s",
    "script_timeout_s": "script_timeout_s",
    "max_worker_attempts": "max_worker_attempts",
    "structured_outputs": "structured_outputs",
    "sdk_timeout_s": "sdk_timeout_s",
    "sandbox": "sandbox",
    "spending_ceiling_usd": "spending_ceiling_usd",
    "datasets": "datasets",
}


def validate_state(plan, state):
    if state.get("comparison_plan_sha256") != plan["plan_sha256"]:
        raise ValueError("Comparison plan mismatch")
    if any(state.get(k) != v for k, v in plan["identity"].items()):
        raise ValueError("Code, controller or rail bundle mismatch")
    for key, pkey in BATCH_SETTINGS.items():
        if state.get(key) != plan[pkey]:
            raise ValueError("Incompatible batch setting: " + key)
    stage = state.get("stage")
    if stage not in {"acceptance", "ablation"} or state.get("rails_enabled") != (
        stage == "acceptance"
    ):
        raise ValueError("Arm rail setting differs from plan")
    if state.get("replicates") != (3 if stage == "acceptance" else 1):
        raise ValueError("Arm replicate count differs from plan")
    planned = {j["study_id"]: j for j in plan["jobs"]}
    seen = set()
    for job in state["jobs"]:
        key = job.get("study_id")
        if (
            key not in planned
            or any(job.get(k) != v for k, v in planned[key].items())
            or job["stage"] != stage
        ):
            raise ValueError("Study identity differs from plan")
        if key in seen:
            raise ValueError("Duplicate study ID: " + key)
        seen.add(key)

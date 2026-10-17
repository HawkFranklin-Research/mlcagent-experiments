"""Prepare an acceptance/ablation plan without inference or deployments."""

import argparse
import json
from pathlib import Path

import run_batch as batch
from comparison import build_plan, runtime_identity
from mlca.paths import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--provider", choices=["openrouter", "deepseek"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high", "max"]
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-limit-mib", type=int)
    parser.add_argument("--deadline-s", type=float, default=5400)
    parser.add_argument("--rpm", type=float, default=60)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Plan output must be a new file")
    identity, datasets = batch.source_identity(), batch.entries()
    evidence, succeeded = [], set()
    for path in sorted(args.development_root.glob("batch-*/batch.json")):
        state = json.loads(path.read_text())
        if state.get("status") != "finished" or not state.get("rails_enabled"):
            continue
        successes = [
            j["dataset"] for j in state["jobs"] if j.get("result", {}).get("success")
        ]
        succeeded.update(successes)
        evidence.append(
            {
                "file": str(path),
                "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                "provider": state.get("provider", "openrouter"),
                "model": state["model"],
                "qualified_tables": successes,
            }
        )
    if not {d["id"] for d in datasets} <= succeeded:
        raise ValueError("Every development table must have qualified at least once")
    plan = build_plan(
        identity,
        datasets,
        provider=args.provider,
        model=args.model,
        effort=args.reasoning_effort,
        threads=args.threads,
        memory_limit_mib=args.memory_limit_mib,
        runtime=runtime_identity(batch.CODE),
        deadline_s=args.deadline_s,
        rpm=args.rpm,
    )
    # Evidence is research metadata; never passed to the agent.
    from comparison import digest

    plan.pop("plan_sha256")
    plan["development_evidence"] = evidence
    plan["plan_sha256"] = digest(plan)
    atomic_json(args.out, plan)
    print(
        json.dumps({"plan_sha256": plan["plan_sha256"], "studies": len(plan["jobs"])})
    )


if __name__ == "__main__":
    main()

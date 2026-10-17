"""Validate and summarize disjoint comparison shards. Never deploys or runs a model."""

import argparse
import json
from pathlib import Path

import run_batch as batch
from comparison import read_plan, validate_state
from mlca.paths import atomic_json


def pool(plan, folders):
    planned = {j["study_id"]: j for j in plan["jobs"]}
    jobs, sources, requests = {}, [], {}
    for folder in folders:
        folder = Path(folder).resolve()
        state = json.loads((folder / "batch.json").read_text())
        validate_state(plan, state)
        for job in state["jobs"]:
            study_id = job.get("study_id")
            if study_id not in planned or any(
                job.get(k) != v for k, v in planned[study_id].items()
            ):
                raise ValueError("Study identity differs from plan")
            if study_id in jobs:
                raise ValueError("Duplicate study ID: " + study_id)
            result = job.get("result", {})
            enforcement = result.get("enforced_resources")
            if enforcement and (
                enforcement.get("threads") != plan["threads"]
                or enforcement.get("memory_limit_mib") != plan["memory_limit_mib"]
                or enforcement.get("memory_enforcement") != plan["memory_enforcement"]
            ):
                raise ValueError("Observed resource enforcement differs from plan")
            jobs[study_id] = {**job, "source_folder": str(folder)}
        source_hash = (
            __import__("hashlib")
            .sha256((folder / "batch.json").read_bytes())
            .hexdigest()
        )
        sources.append(
            {
                "folder": str(folder),
                "batch_sha256": source_hash,
                "machine": state.get("machine"),
                "concurrency": state.get("concurrency"),
                "rpm": state.get("rpm"),
            }
        )
        for record in batch.request_records(folder):
            key = record["request_id"]
            if key in requests:
                raise ValueError("Duplicate request ID across shards: " + key)
            requests[key] = record
    missing = sorted(set(planned) - set(jobs))
    arms = {}
    for stage in ["acceptance", "ablation"]:
        expected = [j for j in plan["jobs"] if j["stage"] == stage]
        present = [j for j in jobs.values() if j["stage"] == stage]
        finished = [j for j in present if j["status"] == "finished"]
        arms[stage] = {
            "planned": len(expected),
            "present": len(present),
            "finished": len(finished),
            "qualified": sum(
                j.get("result", {}).get("success") is True for j in finished
            ),
            "failed": sum(
                j.get("result", {}).get("success") is not True for j in finished
            ),
            "complete": len(finished) == len(expected),
        }
    acceptance = arms["acceptance"]
    candidate = (
        "pending"
        if not acceptance["complete"]
        else "accepted"
        if acceptance["qualified"] == acceptance["planned"]
        else "not_accepted"
    )
    pairs = []
    for planned_job in plan["jobs"]:
        if planned_job["stage"] != "ablation":
            continue
        a, b = (
            jobs.get(planned_job["paired_study_id"]),
            jobs.get(planned_job["study_id"]),
        )
        splits = [
            j.get("result", {}).get("split_parameters") if j else None for j in [a, b]
        ]
        hashes = [
            j.get("result", {}).get("split_sha256") if j else None for j in [a, b]
        ]
        pairs.append(
            {
                "acceptance": planned_job["paired_study_id"],
                "ablation": planned_job["study_id"],
                "both_finished": bool(
                    a and b and a["status"] == b["status"] == "finished"
                ),
                "protocol_seed": planned_job["protocol_seed"],
                "split_parameters_match": splits[0] == splits[1]
                if all(s is not None for s in splits)
                else None,
                "actual_split_parameters": splits,
                "split_membership_match": hashes[0] == hashes[1]
                if all(h is not None for h in hashes)
                else None,
                "actual_split_sha256": hashes,
            }
        )
    recorded_requests = sum(
        j.get("result", {}).get("usage", {}).get("requests", 0) for j in jobs.values()
    )
    accounting_complete = len(requests) == recorded_requests and all(
        "usage" in j.get("result", {})
        for j in jobs.values()
        if j["status"] == "finished"
    )
    return {
        "schema_version": "1",
        "comparison_plan_sha256": plan["plan_sha256"],
        "candidate_acceptance": candidate,
        "arms": arms,
        "missing_studies": missing,
        "jobs": [jobs[key] for key in sorted(jobs)],
        "resource_evidence_unavailable": sorted(
            key
            for key, job in jobs.items()
            if job["status"] == "finished"
            and not job.get("result", {}).get("enforced_resources")
        ),
        "pairs": pairs,
        "sources": sources,
        "accounting": batch.accounting(list(requests.values()))
        if accounting_complete
        else None,
        "accounting_complete": accounting_complete,
        "accounting_note": "Raw request records required for recomputed totals; missing records are unavailable, never zero.",
        "timing_caveats": [
            "Host hardware, concurrent load and node-local pacing may differ; do not attribute wall-time differences solely to rails.",
            *(
                [
                    "No per-study RAM ceiling enforced; memory-sensitive comparisons require matching resource conditions."
                ]
                if plan["memory_limit_mib"] is None
                else []
            ),
        ],
        "scope": "Same harness with instruction rails on/off; failures retained even if candidate is not accepted. Not a bare-LLM comparator.",
    }


def write_summary(out, result):
    atomic_json(out / "pooled.json", result)
    lines = [
        "# Acceptance / instruction-rails ablation",
        "",
        "Candidate acceptance: " + result["candidate_acceptance"],
        "",
        "| Arm | Planned | Finished | Qualified | Failed |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, stats in result["arms"].items():
        lines.append(
            f"| {arm} | {stats['planned']} | {stats['finished']} | {stats['qualified']} | {stats['failed']} |"
        )
    lines += [
        "",
        "Missing studies: " + str(len(result["missing_studies"])),
        "",
        result["scope"],
        "",
        *result["timing_caveats"],
        "",
        "This pool references original folders; it does not copy private data, credentials or projects.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("batches", type=Path, nargs="+")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Choose a fresh pool output directory")
    result = pool(read_plan(args.plan), args.batches)
    args.out.mkdir(parents=True)
    write_summary(args.out, result)


if __name__ == "__main__":
    main()

"""Cost per phase and automatic recovery per set, from existing run records (no model calls).

Cost: every request record (.mlca/requests/*.json and usage.jsonl, deduplicated by request id). DeepSeek
API requests are priced at the published deepseek-flash rate (see analysis/eval02/collect.py); OpenRouter
requests (development batches 1-8) use the cost OpenRouter returned with the response, and requests
without one are counted as unpriced. Retries and failed attempts are included.

Recovery (R2-2): per step, from the event log: completed at the first attempt, recovered (completed after
at least one failed attempt, i.e. by MLCA's own repair), or failed.
Outputs: phases.csv (per set), recovery.csv (per set), SUMMARY.md.
"""

import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "eval02"))
from collect import price  # noqa: E402  (same DeepSeek pricing as the eval-02 analysis)

SETS = [  # set, phase, folders (as in the paper's results ledger, run_sets.csv)
    ("dev-b01", "development", ["bench/dev-loop/batch-01"]), ("dev-b02", "development", ["bench/dev-loop/batch-02"]),
    ("dev-b03", "development", ["bench/dev-loop/batch-03"]), ("dev-b04", "development", ["bench/dev-loop/batch-04"]),
    ("dev-b05", "development", ["bench/dev-loop/batch-05"]), ("dev-b06", "development", ["bench/dev-loop/batch-06"]),
    ("dev-b07", "development", ["bench/dev-loop/batch-07"]), ("dev-b08", "development", ["bench/dev-loop/batch-08"]),
    ("acc-01", "acceptance", ["bench/comparison/acceptance"]), ("dev-b09", "development", ["bench/dev-loop/batch-09"]),
    ("acc-02", "acceptance", ["bench/comparison/acceptance-02"]), ("dev-b10", "development", ["bench/dev-loop/batch-10"]),
    ("dev-b11", "development", ["bench/dev-loop/batch-11"]), ("dev-b12", "development", ["bench/dev-loop/batch-12"]),
    ("dev-b13", "development", ["bench/dev-loop/batch-13"]), ("heldout-01", "heldout (rails 0.13)", ["bench/heldout/heldout-01"]),
    ("eval-01", "evaluation (rails 0.13)", ["bio-bench/runs/deepseek-flash/eval-01"]),
    ("v2-b01", "development (after reopening)", ["bench/dev-loop-v2/batch-01"]),
    ("v2-b02", "development (after reopening)", ["bench/dev-loop-v2/batch-02"]),
    ("eval-02", "evaluation (released)", [f"bio-bench/runs/deepseek-flash/{f}" for f in
                                          ["eval-02", "eval-02-vm1", "eval-02-vm1b", "eval-02-vm2", "eval-02-vm3"]]),
    ("heldout-02", "heldout (released)", ["bench/heldout/heldout-02-vm1", "bench/heldout/heldout-02-vm2"]),
]


def requests(folder):
    seen = {}
    for f in glob.glob(str(REPO / folder / "**/.mlca/requests/*.json"), recursive=True):
        try:
            u = json.load(open(f)).get("usage") or {}
        except (json.JSONDecodeError, OSError):
            continue
        if u.get("request_id"):
            seen[u["request_id"]] = u
    for f in glob.glob(str(REPO / folder / "**/.mlca/usage.jsonl"), recursive=True):
        for line in open(f):
            if line.strip():
                u = json.loads(line)
                seen.setdefault(u["request_id"], u)
    return seen


def cost(u):
    if u.get("provider_kind") == "deepseek" and u.get("ts"):
        return price(u)
    return u.get("cost_usd")


def recovery(folder):
    counts = defaultdict(int)
    for log in glob.glob(str(REPO / folder / "**/run-001/.mlca/log/events.jsonl"), recursive=True):
        failed_attempts = defaultdict(int)
        for line in open(log):
            e = json.loads(line)
            step = e.get("step_id")
            if e["type"] == "RunFinished" and (e.get("payload") or {}).get("category"):
                failed_attempts[step] += 1
            elif e["type"] == "StepCompleted":
                counts["recovered" if failed_attempts[step] else "first_try"] += 1
            elif e["type"] == "StepFailed":
                counts["failed"] += 1
    return counts


def main():
    phase_rows, recovery_rows = [], []
    for sid, phase, folders in SETS:
        reqs = {}
        for f in folders:
            reqs.update(requests(f))
        priced = [cost(u) for u in reqs.values()]
        known = [c for c in priced if c is not None]
        tokens = lambda k: sum(u.get(k) or 0 for u in reqs.values())  # noqa: E731
        phase_rows.append({
            "set": sid, "phase": phase, "requests": len(reqs),
            "providers": "|".join(sorted({u.get("provider_kind") or "?" for u in reqs.values()})),
            "input_tokens": tokens("input_tokens"), "cached_tokens": tokens("cached_tokens"),
            "output_tokens": tokens("output_tokens"), "reasoning_tokens": tokens("reasoning_tokens"),
            "cost_usd_known": round(sum(known), 2), "requests_unpriced": len(priced) - len(known),
        })
        c = defaultdict(int)
        for f in folders:
            for k, v in recovery(f).items():
                c[k] += v
        total = c["first_try"] + c["recovered"] + c["failed"]
        needing = c["recovered"] + c["failed"]
        recovery_rows.append({
            "set": sid, "phase": phase, "steps": total, "first_try": c["first_try"], "recovered": c["recovered"],
            "failed": c["failed"], "steps_needing_repair": needing,
            "recovery_rate": round(c["recovered"] / needing, 3) if needing else None,
        })
    for name, rows in [("phases", phase_rows), ("recovery", recovery_rows)]:
        with open(HERE / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    by_phase = defaultdict(lambda: defaultdict(float))
    for r in phase_rows:
        for k in ["requests", "input_tokens", "output_tokens", "cost_usd_known", "requests_unpriced"]:
            by_phase[r["phase"]][k] += r[k]
    lines = ["# Cost per phase and automatic recovery (generated by phases.py)", "",
             "| Phase | Requests | Input tokens | Output tokens | Cost US$ (priced requests) | Unpriced requests |",
             "|---|---:|---:|---:|---:|---:|"]
    for phase, v in by_phase.items():
        lines.append(f"| {phase} | {int(v['requests'])} | {int(v['input_tokens'])} | {int(v['output_tokens'])} | "
                     f"{v['cost_usd_known']:.2f} | {int(v['requests_unpriced'])} |")
    total_cost = sum(v["cost_usd_known"] for v in by_phase.values())
    lines += ["", f"Whole study, priced requests: **US$ {total_cost:.2f}**.", "",
              "| Set | Steps | First try | Recovered by repair | Failed | Recovery rate |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['set']} | {r['steps']} | {r['first_try']} | {r['recovered']} | {r['failed']} | {r['recovery_rate']} |"
              for r in recovery_rows]
    (HERE / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

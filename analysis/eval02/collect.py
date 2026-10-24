"""Read the eval-02 records (no model calls) into tables for the paper's answers.

Outputs (in this folder): studies.csv (one row per official study, 27), decisions.csv (the agent's
setup and modelling decisions per study), attempts_usage.csv (every attempt including infrastructure
retries), egress_summary.csv, cost.json and SUMMARY.md.

Prices: DeepSeek API pricing page (api-docs.deepseek.com/quick_start/pricing), read 24 October 2026,
model deepseek-flash (DeepSeek-V4.1-Flash-1017), USD per 1M tokens, peak: cache hit 0.006, cache miss
0.30, output 1.20; off-peak (outside 01-04 and 06-10 UTC, Mon-Fri) half. Output tokens include
reasoning tokens (reasoning is reported as a subset).
"""

import csv
import glob
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "bio-bench/runs/deepseek-flash"
FOLDERS = ["eval-02", "eval-02-vm1", "eval-02-vm1b", "eval-02-vm2", "eval-02-vm3"]
RECORDS = ["eval-02-vm-envfail", "_envfail-py312-2026-10-23"]  # infrastructure records
# Where each official study's final attempt lives (laptop connection failures were rerun on vm1/vm1b).
OFFICIAL = {
    "s1_anemia": "eval-02", "s2_rna_er": "eval-02", "s2_rna_pr": "eval-02",
    "s2_rna_her2": "eval-02-vm1", "s2_proteome_her2": "eval-02-vm2", "s2_proteome_pr": "eval-02-vm2",
    "s3_pam50": "eval-02-vm3", "s4_hnscc": "eval-02-vm3",
}
PEAK = {"hit": 0.006, "miss": 0.30, "out": 1.20}
ESTIMATORS = re.compile(
    r"\b(LogisticRegression\w*|RandomForest\w+|ExtraTrees\w+|GradientBoosting\w+|HistGradientBoosting\w+|"
    r"LinearSVC|SVC|SVR|KNeighbors\w+|DecisionTree\w+|XGB\w+|LGBM\w+|Ridge\w*|Lasso\w*|ElasticNet\w*|"
    r"LinearRegression|CoxPH\w*|CoxnetSurvival\w*|RandomSurvivalForest|GaussianNB|MLP\w+|SGD\w+|"
    r"KMeans|PCA|TruncatedSVD|SelectKBest|VarianceThreshold|SelectFromModel|RFE\w*)\b"
)


def official_folder(dataset, replicate):
    if dataset == "s2_proteome_er":
        return "eval-02" if replicate == 3 else "eval-02-vm1b"
    return OFFICIAL[dataset]


def jsonl(path):
    try:
        return [json.loads(line) for line in open(path) if line.strip()]
    except FileNotFoundError:
        return []


def price(record):
    ts = datetime.fromisoformat(record["ts"])
    peak = ts.weekday() < 5 and (1 <= ts.hour < 4 or 6 <= ts.hour < 10)
    factor = 1.0 if peak else 0.5
    inp = record.get("input_tokens") or 0
    hit = record.get("cached_tokens") or 0
    out = record.get("output_tokens") or 0
    return factor * ((inp - hit) * PEAK["miss"] + hit * PEAK["hit"] + out * PEAK["out"]) / 1e6


def attempt_usage(attempt):
    seen = {}
    for f in glob.glob(f"{attempt}/*/.mlca/usage.jsonl"):
        for r in jsonl(f):
            seen[r["request_id"]] = r
    return seen


def egress(attempt):
    rows = []
    for f in glob.glob(f"{attempt}/*/.mlca/egress.jsonl"):
        rows += jsonl(f)
    return rows


def decisions(project):
    state = json.load(open(project / ".mlca/state.json"))
    events = jsonl(project / ".mlca/log/events.jsonl")
    task = state.get("task") or {}
    split = (state.get("split") or {}).get("parameters") or {}
    out = {
        "problem_type": task.get("type"),
        "targets": "|".join(task.get("targets") or []),
        "primary_metric": task.get("primary_metric"),
        "excluded_columns": "|".join(state.get("excluded_columns") or []),
        "group_column": state.get("group_column") or "",
        "time_columns": "|".join(task.get("time_columns") or []),
        "data_decisions": "|".join(d.get("effect", {}).get("kind", "") for d in state.get("data_decisions") or []),
        "split_method": split.get("method", ""),
        "split_seed": split.get("seed", ""),
        "split_fractions": f"{split.get('train_fraction', '')}/{split.get('val_fraction', '')}",
    }
    models, n_features, test_metrics = [], [], []
    for e in events:
        if e["type"] != "StepCompleted":
            continue
        p = e["payload"]
        step = p.get("step_id", "")
        if "_model_" in step:
            n_features += [len(f["columns"]) for f in p.get("fits", [])]
            script = project / ".mlca/steps" / step / "script.py"
            if script.exists():
                models += [m for m in ESTIMATORS.findall(script.read_text()) if m not in ("SelectKBest", "VarianceThreshold", "SelectFromModel", "PCA", "TruncatedSVD") and not m.startswith("RFE")]
        test_metrics += [f"{m['name']}={m['value']:.4f}" for m in p.get("metrics", []) if m.get("split") == "test" and m.get("value") is not None]
    out["model_estimators"] = "|".join(sorted(set(models)))
    out["features_fitted"] = "|".join(str(n) for n in n_features)
    out["test_metrics"] = "|".join(test_metrics)
    return out


def main():
    studies, decision_rows, attempt_rows, egress_rows = [], [], [], []
    all_requests = {}
    for folder in FOLDERS + RECORDS:
        for batch in glob.glob(str(RUNS / folder / "**/batch.json"), recursive=True):
            root = Path(batch).parent
            state = json.load(open(batch))
            for job in state["jobs"]:
                for i, relative in enumerate(job.get("attempts", [])):
                    attempt = root / relative
                    usage = attempt_usage(attempt)
                    all_requests.update(usage)
                    final = i == len(job["attempts"]) - 1
                    result = (job.get("result") or {}) if final else next(
                        (r.get("prior_result") or {} for r in job.get("retries", []) if r.get("retry_of") == relative), {})
                    attempt_rows.append({
                        "folder": str(root.relative_to(RUNS)), "study": job["id"], "attempt": Path(relative).name,
                        "record_only": folder in RECORDS, "category": result.get("category", ""),
                        "success": result.get("success"), "requests": len(usage),
                        "cost_usd": round(sum(price(r) for r in usage.values() if r.get("ts")), 4),
                    })
                if folder in RECORDS or root.name != folder:
                    continue
                if official_folder(job["dataset"], job["replicate"]) != folder:
                    continue
                attempt = root / job["attempts"][-1]
                usage = attempt_usage(attempt)
                res = job.get("result") or {}
                inp = sum(r.get("input_tokens") or 0 for r in usage.values())
                row = {
                    "task": job["dataset"], "run": job["replicate"], "folder": folder,
                    "attempt": Path(job["attempts"][-1]).name, "success": res.get("success"),
                    "category": res.get("category", ""), "furthest_event": res.get("furthest_event", ""),
                    "requests": len(usage), "input_tokens": inp,
                    "cached_tokens": sum(r.get("cached_tokens") or 0 for r in usage.values()),
                    "output_tokens": sum(r.get("output_tokens") or 0 for r in usage.values()),
                    "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in usage.values()),
                    "model_time_s": round(sum((r.get("latency_ms") or 0) for r in usage.values()) / 1000, 1),
                    "wall_s": res.get("wall_s"), "cost_usd": round(sum(price(r) for r in usage.values() if r.get("ts")), 4),
                    "host": (res.get("host") or {}).get("hostname", ""),
                }
                project = attempt / "run-001"
                step_wall = 0.0
                for e in jsonl(project / ".mlca/log/events.jsonl"):
                    if e["type"] == "StepCompleted":
                        step_wall += e["payload"].get("wall_time_s") or 0
                row["script_time_s"] = round(step_wall, 1)
                studies.append(row)
                if (project / ".mlca/state.json").exists():
                    decision_rows.append({"task": row["task"], "run": row["run"], "success": row["success"], **decisions(project)})
                eg = egress(attempt)
                cats = Counter(c for r in eg for c in r.get("categories", []))
                by_role = defaultdict(int)
                for r in eg:
                    by_role[r.get("role", "")] += r.get("bytes_sent") or 0
                egress_rows.append({
                    "task": row["task"], "run": row["run"], "requests": len(eg),
                    "bytes_total": sum(by_role.values()), "bytes_navigator": by_role.get("navigator", 0),
                    "bytes_worker": by_role.get("worker", 0), "max_request_bytes": max([r.get("bytes_sent") or 0 for r in eg] or [0]),
                    "categories": "|".join(f"{k}:{v}" for k, v in sorted(cats.items())),
                })
    studies.sort(key=lambda r: (r["task"], r["run"]))
    decision_rows.sort(key=lambda r: (r["task"], r["run"]))
    egress_rows.sort(key=lambda r: (r["task"], r["run"]))
    for name, rows in [("studies", studies), ("decisions", decision_rows), ("attempts_usage", attempt_rows), ("egress_summary", egress_rows)]:
        with open(HERE / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    official_cost = sum(r["cost_usd"] for r in studies)
    every_request = sum(price(r) for r in all_requests.values() if r.get("ts"))
    cost = {
        "pricing": "deepseek-flash (DeepSeek-V4.1-Flash-1017), USD/1M: peak hit 0.006, miss 0.30, output 1.20; off-peak half",
        "official_27_studies_usd": round(official_cost, 2),
        "per_successful_study_usd_median": round(statistics.median([r["cost_usd"] for r in studies if r["success"]]), 3),
        "all_eval02_attempts_including_infrastructure_usd": round(every_request, 2),
        "requests_official": sum(r["requests"] for r in studies),
        "requests_all_attempts": len(all_requests),
        "tokens_official": {k: sum(r[k] for r in studies) for k in ["input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"]},
    }
    (HERE / "cost.json").write_text(json.dumps(cost, indent=2) + "\n")
    print(json.dumps(cost, indent=2))
    print(f"studies={len(studies)} successful={sum(1 for r in studies if r['success'])}")


if __name__ == "__main__":
    main()

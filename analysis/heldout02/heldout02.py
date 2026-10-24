"""Held-out tables on the released version (heldout-02): the agent's setup decisions per study, and paired
bootstrap intervals of AutoML minus MLCA on the shared test rows (no model calls, no refits).
Bank Marketing: ROC AUC. Wine Quality: RMSE (lower is better; MLCA's test predictions are used only if they
are numeric). Steel has no MLCA metric (multi-label framing); Wholesale has no AutoML counterpart.
Outputs: decisions.csv, bootstrap.csv.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "eval02"))
from collect import decisions  # noqa: E402

FOLDERS = [REPO / "bench/heldout/heldout-02-vm1", REPO / "bench/heldout/heldout-02-vm2"]
AUTOML = REPO / "bench/automl-heldout-02"
SPEC = REPO / "automl/specs/heldout.yaml"
MLCA_PY = REPO.parent / "mlca-revized/.venv/bin/python"
SCRATCH = Path(os.environ.get("MLCA_BOOT_SCRATCH", Path.home() / ".cache/mlca-heldout02-bootstrap"))


def decision_rows():
    rows = []
    for folder in FOLDERS:
        state = json.load(open(folder / "batch.json"))
        for job in state["jobs"]:
            res = job.get("result") or {}
            project = folder / job["attempts"][-1] / "run-001"
            proposal = {}
            log = project / ".mlca/log/events.jsonl"
            if log.exists():
                for line in open(log):
                    e = json.loads(line)
                    if e["type"] == "InferenceRecorded":
                        proposal = e["payload"]["proposal"]
            row = {"table": job["dataset"], "run": job["replicate"], "success": res.get("success"),
                   "category": res.get("category", ""), "furthest_event": res.get("furthest_event", ""),
                   "inferred_problem_type": proposal.get("problem_type"),
                   "inferred_targets": "|".join(proposal.get("targets") or [])}
            if (project / ".mlca/state.json").exists():
                row.update({k: v for k, v in decisions(project).items() if k in
                            ("split_method", "model_estimators", "features_fitted", "test_metrics")})
            rows.append(row)
    return sorted(rows, key=lambda r: (r["table"], r["run"]))


def mlca_predictions(task, run_label, project):
    work = SCRATCH / task / run_label
    out = work / "mlca_test_predictions.parquet"
    if not out.exists():
        work.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(MLCA_PY), str(REPO / "automl/export_split.py"), "--spec", str(SPEC), "--task", task,
                        "--project", str(project), "--work", str(work), "--run-label", run_label], check=True,
                       capture_output=True, env={**os.environ, "PYTHONPATH": str(REPO.parent / "mlca-revized/src")})
    return pd.read_parquet(out).set_index("row_id")


def bootstrap_rows():
    rng = np.random.default_rng(20261024)
    rows = []
    for result in sorted(AUTOML.glob("*/*/*/*/result.json")):
        r = json.loads(result.read_text())
        if r.get("status") != "completed" or r["task"] not in ("bank_marketing", "wine_quality"):
            continue
        study, attempt = r["run_label"].split("/")
        project = next(f / "studies" / study / attempt / "run-001" for f in FOLDERS
                       if (f / "studies" / study / attempt / "run-001").exists())
        theirs = pd.read_parquet(result.parent / "test_predictions.parquet").set_index("row_id")
        mine = mlca_predictions(r["task"], r["run_label"], project).loc[theirs.index]
        y = theirs["label"].to_numpy()
        if r["task"] == "bank_marketing":
            classes = list(r["classes"])
            p_auto = theirs["proba_1"].to_numpy()
            p_mlca = mine[f"proba__{classes[1]}"].to_numpy()

            def metric(i):
                return roc_auc_score(y[i], p_auto[i]) - roc_auc_score(y[i], p_mlca[i])

            name, a_mlca, a_auto = "roc_auc", roc_auc_score(y, p_mlca), roc_auc_score(y, p_auto)
        else:
            column = next((c for c in mine.columns if c.startswith("y_pred") or c == "prediction"), None)
            pred = pd.to_numeric(mine[column], errors="coerce").to_numpy() if column else None
            if pred is None or np.isnan(pred).any():
                continue
            p_auto = theirs["prediction"].to_numpy()

            def metric(i):
                return mean_squared_error(y[i], p_auto[i]) ** 0.5 - mean_squared_error(y[i], pred[i]) ** 0.5

            name = "rmse"
            a_mlca, a_auto = mean_squared_error(y, pred) ** 0.5, mean_squared_error(y, p_auto) ** 0.5
        diffs = []
        while len(diffs) < 2000:
            i = rng.integers(0, len(y), len(y))
            if name == "roc_auc" and len(set(y[i])) < 2:
                continue
            diffs.append(metric(i))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"table": r["task"], "run": r["run_label"], "framework": r["framework"], "metric": name,
                     "n_test": len(y), "mlca": round(a_mlca, 4), "automl": round(a_auto, 4),
                     "diff_automl_minus_mlca": round(a_auto - a_mlca, 4), "ci95_low": round(float(lo), 4),
                     "ci95_high": round(float(hi), 4), "interval_excludes_zero": bool(lo > 0 or hi < 0)})
        print(rows[-1], flush=True)
    return rows


def main():
    for name, rows in [("decisions", decision_rows()), ("bootstrap", bootstrap_rows())]:
        with open(HERE / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()

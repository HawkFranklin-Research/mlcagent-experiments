"""Paired bootstrap intervals for the biomedical AutoML comparison (no model calls, no refits).

For every completed pair in bench/automl-bio, MLCA's saved test predictions are re-exported from the
run (automl/export_split.py, hash-verified) into a scratch folder, and AutoML's saved test predictions
are read from its result folder. Both are scored on the same test rows (ROC AUC; one-vs-rest macro for
PAM50) and the difference AutoML minus MLCA gets a 95% percentile interval from 2,000 paired bootstrap
resamples of the test rows (resamples with a single class are redrawn).
Usage: ../mlca-revized/.venv/bin/python analysis/automl_bootstrap/bootstrap.py
"""

import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "bench/automl-bio"
RUNS = REPO / "bio-bench/runs/deepseek-flash"
SPEC = REPO / "automl/specs/biomedical.yaml"
MLCA_PY = REPO.parent / "mlca-revized/.venv/bin/python"
SCRATCH = Path(os.environ.get("MLCA_BOOT_SCRATCH", Path.home() / ".cache/mlca-automl-bootstrap"))
FOLDERS = ["eval-02", "eval-02-vm1", "eval-02-vm1b", "eval-02-vm2", "eval-02-vm3"]


def project_for(task, run_label):
    study, attempt = run_label.split("/")
    for folder in FOLDERS:
        project = RUNS / folder / "studies" / study / attempt / "run-001"
        if project.exists():
            return project
    raise FileNotFoundError(run_label)


def auc(y, p, classes):
    if len(classes) == 2:
        return roc_auc_score(y, p[:, 1])
    return roc_auc_score(y, p, multi_class="ovr", average="macro", labels=classes)


def main():
    rng = np.random.default_rng(20261024)
    rows = []
    for result in sorted(RESULTS.glob("*/*/*/*/result.json")):
        r = json.loads(result.read_text())
        if r.get("status") != "completed":
            continue
        task, framework, run_label = r["task"], r["framework"], r["run_label"]
        work = SCRATCH / task / run_label
        mlca_file = work / "mlca_test_predictions.parquet"
        if not mlca_file.exists():
            work.mkdir(parents=True, exist_ok=True)
            subprocess.run([str(MLCA_PY), str(REPO / "automl/export_split.py"), "--spec", str(SPEC), "--task", task,
                            "--project", str(project_for(task, run_label)), "--work", str(work), "--run-label", run_label],
                           check=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(REPO.parent / "mlca-revized/src")})
        mine = pd.read_parquet(mlca_file).set_index("row_id")
        theirs = pd.read_parquet(result.parent / "test_predictions.parquet").set_index("row_id")
        classes = list(r["classes"])
        mine = mine.loc[theirs.index]
        y = theirs["label"].to_numpy()
        p_auto = theirs[[f"proba_{i}" for i in range(len(classes))]].to_numpy()
        p_mlca = mine[[f"proba__{c}" for c in classes]].to_numpy()
        a_mlca, a_auto = auc(y, p_mlca, classes), auc(y, p_auto, classes)
        diffs = []
        while len(diffs) < 2000:
            i = rng.integers(0, len(y), len(y))
            if len(set(y[i])) < len(classes):
                continue
            diffs.append(auc(y[i], p_auto[i], classes) - auc(y[i], p_mlca[i], classes))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"task": task, "run": run_label, "framework": framework, "n_test": len(y),
                     "auc_mlca": round(a_mlca, 4), "auc_automl": round(a_auto, 4),
                     "diff_automl_minus_mlca": round(a_auto - a_mlca, 4),
                     "ci95_low": round(float(lo), 4), "ci95_high": round(float(hi), 4),
                     "interval_excludes_zero": bool(lo > 0 or hi < 0)})
        print(rows[-1], flush=True)
    with open(HERE / "bootstrap.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

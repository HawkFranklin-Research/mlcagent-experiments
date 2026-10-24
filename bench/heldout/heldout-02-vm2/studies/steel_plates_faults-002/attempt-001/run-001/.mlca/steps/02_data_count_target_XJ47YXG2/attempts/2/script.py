import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def _load_original():
    original = next(Path(".").glob("*.py"), None)
    if original is None:
        raise FileNotFoundError("No original script found in working directory.")

    spec = importlib.util.spec_from_file_location("_original_mlca", original)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_original_mlca"] = module
    spec.loader.exec_module(module)
    return module


def run() -> None:
    module = _load_original()

    mlca_runtime = module.mlca_runtime

    inputs = mlca_runtime.inputs()
    protocol = json.loads(inputs["_protocol"].read_text())
    targets = protocol["targets"]

    df = pd.read_parquet(inputs["steel_plates_faults_uci"])

    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise KeyError(
            f"Missing target columns: {missing}. Available columns: {list(df.columns)}"
        )

    n_rows = len(df)
    rows = []
    for t in targets:
        n = int((df[t] == 1).sum())
        share = round(n / n_rows, 4) if n_rows else float("nan")
        rows.append({"class": t, "n": n, "share": share})
        mlca_runtime.record_metric(f"{t}_share", share, "external")

    counts = pd.DataFrame(rows)[["class", "n", "share"]]
    mlca_runtime.save_table(
        counts, "target_counts.csv", caption="Per-class counts and shares"
    )

    flagged = [r["class"] for r in rows if r["share"] < 0.05]
    print(f"rows: {n_rows}")
    print(f"targets: {len(targets)}")
    print(f"under 5%: {flagged if flagged else 'none'}")


if __name__ == "__main__":
    run()

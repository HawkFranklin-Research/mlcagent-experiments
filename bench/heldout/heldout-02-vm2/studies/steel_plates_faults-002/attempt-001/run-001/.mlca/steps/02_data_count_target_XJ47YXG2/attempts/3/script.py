import mlca_runtime
import json
import pandas as pd


def run() -> None:
    mlca_runtime.set_seed(42)

    inputs = mlca_runtime.inputs()
    protocol = json.loads(inputs["_protocol"].read_text())
    if "targets" not in protocol:
        raise KeyError(
            f"_protocol missing 'targets'; available keys: {list(protocol.keys())}"
        )

    targets = list(protocol["targets"])
    df = pd.read_parquet(inputs["steel_plates_faults_uci"])

    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise KeyError(
            f"Missing target columns: {missing}; available columns: {list(df.columns)}"
        )

    n_rows = len(df)
    rows = []
    flagged = []
    for t in targets:
        n = int((df[t] == 1).sum())
        share = float(n / n_rows) if n_rows else float("nan")
        rows.append({"class": t, "n": n, "share": round(share, 4)})
        if share < 0.05:
            flagged.append(t)
        mlca_runtime.record_metric(f"{t}_share", share, "external")

    counts = pd.DataFrame(rows, columns=["class", "n", "share"])
    mlca_runtime.save_table(
        counts, "target_counts.csv", caption="Per-class counts and shares"
    )
    mlca_runtime.record_metric("n_rows", float(n_rows), "external")

    print(f"rows: {n_rows}")
    print(f"targets: {len(targets)}")
    print(f"flagged_under_5pct: {flagged if flagged else 'none'}")


if __name__ == "__main__":
    run()

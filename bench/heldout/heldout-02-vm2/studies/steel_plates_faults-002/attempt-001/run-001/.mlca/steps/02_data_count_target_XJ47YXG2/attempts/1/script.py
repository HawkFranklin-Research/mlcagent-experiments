import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
df = pd.read_parquet(inputs["steel_plates_faults_uci"])

missing = [t for t in targets if t not in df.columns]
if missing:
    raise KeyError(f"Missing target columns: {missing}. Available columns: {list(df.columns)}")

n_rows = len(df)
rows = []
for t in targets:
    n = int((df[t] == 1).sum())
    share = round(n / n_rows, 4)
    rows.append({"class": t, "n": n, "share": share, "under_5pct": share < 0.05})
    mlca_runtime.record_metric(f"{t}_share", share, "all")

counts = pd.DataFrame(rows)[["class", "n", "share"]]
mlca_runtime.save_table(counts, "target_counts.csv", caption="Per-class counts and shares")

flagged = [r["class"] for r in rows if r["under_5pct"]]
print(f"rows: {n_rows}")
print(f"targets: {len(targets)}")
print(f"under 5%: {flagged if flagged else 'none'}")

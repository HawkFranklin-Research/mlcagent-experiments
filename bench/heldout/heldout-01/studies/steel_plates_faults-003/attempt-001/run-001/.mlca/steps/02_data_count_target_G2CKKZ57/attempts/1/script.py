import mlca_runtime, json, pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]

df = pd.read_parquet(inputs["steel_plates_faults_uci"])
n_total = len(df)

rows = []
for t in targets:
    n = int(df[t].sum())
    share = n / n_total
    rows.append({"class": t, "n": n, "share": round(share, 4), "under_5pct": bool(share < 0.05)})

out = pd.DataFrame(rows)
mlca_runtime.save_table(out, "target_counts.csv",
                        caption="Counts and row share per target class; flagged if under 5% of rows")
flagged = int(out["under_5pct"].sum())
mlca_runtime.record_metric("n_flagged_classes", flagged, "train")
print(f"rows={n_total} targets={len(targets)}")
print(f"flagged_under_5pct={flagged}")

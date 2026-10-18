import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
df = pd.read_parquet(inputs["anuran_calls_uci"])

target = targets[0]
counts = df[target].value_counts(dropna=False).rename_axis("class").reset_index(name="n")
counts["share"] = counts["n"] / len(df)
counts["flagged_under_5pct"] = counts["share"] < 0.05

mlca_runtime.save_table(counts, "target_counts.csv", caption="Per-class counts and share for target.")

n_flag = int(counts["flagged_under_5pct"].sum())
mlca_runtime.record_metric("n_classes_under_5pct", n_flag, "train")
print(f"rows={len(df)} classes={len(counts)} flagged_under_5pct={n_flag}")

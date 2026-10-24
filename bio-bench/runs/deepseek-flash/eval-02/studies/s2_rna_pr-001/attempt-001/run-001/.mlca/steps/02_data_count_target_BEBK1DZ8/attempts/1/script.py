import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
task = protocol["task"]
assert len(targets) == 1, f"expected one target, got {targets}"
target = targets[0]

df = pd.read_parquet(inputs["rna_pr"])
if target not in df.columns:
    raise KeyError(f"target '{target}' not in columns; sample: {list(df.columns)[:10]}")

counts = df[target].value_counts(dropna=False).rename_axis("class").reset_index(name="n")
counts["share"] = counts["n"] / len(df)
counts["flag_under_5pct"] = counts["share"] < 0.05
counts["share"] = counts["share"].round(4)

mlca_runtime.save_table(counts, "target_counts.csv",
                        caption="Target class counts and shares; classes under 5% flagged.")
for _, r in counts.iterrows():
    mlca_runtime.record_metric(f"share_{r['class']}", float(r["share"]), split="all")
print(f"target={target} rows={len(df)} classes={len(counts)} under5={int(counts['flag_under_5pct'].sum())}")

import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
if "targets" not in protocol:
    raise ValueError(f"Missing _protocol['targets']; available top-level keys: {sorted(protocol.keys())}")
targets = protocol["targets"]
if not isinstance(targets, list) or len(targets) != 1:
    raise ValueError(f"Expected exactly one target column, got {targets!r}")
target_col = targets[0]

df = pd.read_parquet(inputs["rna_her2"])
if target_col not in df.columns:
    raise KeyError(f"Target column {target_col!r} not in data columns: {list(df.columns)[:20]}")
if len(df) == 0:
    raise ValueError("No rows in rna_her2")

counts = df[target_col].dropna().value_counts().rename_axis("class").reset_index(name="n")
raw_share = counts["n"] / len(df)
counts["share"] = raw_share.map(lambda v: float(f"{v:.4g}"))
flagged = counts.loc[raw_share < 0.05, "class"].tolist()

mlca_runtime.save_table(counts, "target_counts.csv", caption="Target class counts and shares; classes under 5% of rows are flagged.")
print(f"rows={len(df)}, target={target_col}, classes={len(counts)}")
print(f"flagged_under_5pct={flagged}")
print("saved target_counts.csv")

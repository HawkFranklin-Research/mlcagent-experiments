import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

if "targets" not in protocol:
    raise KeyError(f"Missing 'targets' in _protocol. Available top-level keys: {sorted(protocol.keys())}")
targets = protocol["targets"]
if not targets:
    raise ValueError("No targets declared in _protocol")
target = targets[0]

df = pd.read_parquet(inputs["discovery"])
if target not in df.columns:
    raise KeyError(f"Target column '{target}' not found in discovery data")

s = df[target].astype("string").str.strip()
placeholder_tokens = {"?", "-", "NA", "N/A", ""}
placeholder_mask = s.isna() | s.isin(placeholder_tokens)
n_placeholders = int(placeholder_mask.sum())

if n_placeholders:
    s = s.mask(placeholder_mask, pd.NA)
    print(f"Placeholders found: {n_placeholders} values in 1 column converted to missing")
else:
    print("No placeholder tokens found in the target column")

counts = s.value_counts(dropna=True)
total = int(counts.sum())

counts_df = (
    counts.rename_axis("class")
    .reset_index(name="n")
    .sort_values("n", ascending=False)
    .reset_index(drop=True)
)
counts_df["share"] = counts_df["n"] / total
counts_df["share"] = counts_df["share"].map(lambda v: float(f"{v:.4g}"))

flagged = counts_df.loc[counts_df["share"] < 0.05, "class"].tolist()

mlca_runtime.save_table(
    counts_df[["class", "n", "share"]],
    "target_counts.csv",
    "Target class counts and shares",
)
mlca_runtime.record_metric("n_flagged_under_5pct", float(len(flagged)), "train")

print(f"Rows read: {len(df)}")
print(f"Target: {target}; non-missing classes: {len(counts_df)}")
print(f"Flagged classes (<5%): {flagged}")

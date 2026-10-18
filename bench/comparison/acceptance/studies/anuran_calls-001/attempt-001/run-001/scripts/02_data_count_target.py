import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def get_protocol(key):
    if key not in protocol:
        raise KeyError(f"_protocol missing {key!r}; available top-level keys: {sorted(protocol.keys())}")
    return protocol[key]

targets = get_protocol("targets")
if not isinstance(targets, list) or len(targets) != 1:
    raise ValueError(f"Expected exactly one target in _protocol['targets'], got {targets!r}")
target = targets[0]

df = pd.read_parquet(inputs["anuran_calls_uci"])
if target not in df.columns:
    raise KeyError(f"Target column {target!r} not found; available columns: {list(df.columns)}")

total = len(df)
counts = df[target].value_counts(dropna=True).sort_values(ascending=False)
counts_df = counts.rename_axis("class").reset_index(name="n")
counts_df["share"] = counts_df["n"] / total if total else 0.0
counts_df["share"] = counts_df["share"].map(lambda v: round(float(v), 4))
under_mask = counts_df["share"] < 0.05
n_under = int(under_mask.sum())

mlca_runtime.save_table(
    counts_df[["class", "n", "share"]],
    "target_counts.csv",
    caption="Target class counts and row shares; classes under 5% flagged",
)

print(f"target={target} rows={total} classes={len(counts_df)}")
print(f"classes under 5%: {n_under}")
print("saved target_counts.csv")

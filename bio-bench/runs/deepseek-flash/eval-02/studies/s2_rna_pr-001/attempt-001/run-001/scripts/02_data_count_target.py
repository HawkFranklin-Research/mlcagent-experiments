import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
target = targets[0]

df = pd.read_parquet(inputs["rna_pr"])
if target not in df.columns:
    raise KeyError(f"target '{target}' not in columns; sample: {list(df.columns)[:10]}")

counts = df[target].value_counts(dropna=False).rename_axis("class").reset_index(name="n")
counts["share"] = (counts["n"] / len(df)).round(4)
under = int((counts["n"] / len(df) < 0.05).sum())

mlca_runtime.save_table(
    counts[["class", "n", "share"]],
    "target_counts.csv",
    caption="Target class counts and shares; classes under 5% flagged.",
)
print(f"target={target} rows={len(df)} classes={len(counts)} under5={under}")

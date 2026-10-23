import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

required = ["targets", "task"]
missing = [key for key in required if key not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {sorted(protocol)}")

targets = protocol["targets"]
if not isinstance(targets, list) or len(targets) != 1:
    raise ValueError(f"Expected exactly one target, got {targets!r}")

target = targets[0]
df = pd.read_parquet(paths["gcm_tumour_types_openml"])

if target not in df.columns:
    raise ValueError(f"Target {target!r} not found in table columns")

total_rows = len(df)
counts = df[target].value_counts(dropna=True)
if counts.empty:
    raise ValueError(f"No non-missing values found for target {target!r}")

out = pd.DataFrame({
    "class": counts.index.astype(str),
    "n": counts.to_numpy(dtype=int),
})
out["share"] = (out["n"] / total_rows).map(lambda value: float(f"{value:.4g}"))
out["low_frequency"] = out["share"] < 0.05
out = out.sort_values("n", ascending=False).reset_index(drop=True)

mlca_runtime.save_table(
    out,
    "target_counts.csv",
    caption="Target class counts and shares; low_frequency flags classes under 5% of rows.",
)

print(f"Target {target}: {total_rows} rows, {len(out)} classes")
print(f"Low-frequency classes (<5%): {int(out['low_frequency'].sum())}")
print("Saved target_counts.csv")

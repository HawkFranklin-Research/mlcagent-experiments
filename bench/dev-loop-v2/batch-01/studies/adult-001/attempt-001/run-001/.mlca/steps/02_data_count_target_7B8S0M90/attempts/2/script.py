import json

import pandas as pd

import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
if not targets:
    raise ValueError("No targets declared in _protocol['targets']")

df = pd.read_parquet(inputs["adult_census_uci"])
target = targets[0]
if target not in df.columns:
    raise KeyError(f"Target {target!r} missing; available: {sorted(df.columns)[:20]}")

counts = df[target].value_counts(dropna=False)
total = int(len(df))
share = counts.to_numpy() / total if total else counts.to_numpy() * 0.0
table = pd.DataFrame({
    "class": counts.index.astype(str),
    "n": counts.to_numpy().astype(int),
    "share": pd.Series(share).round(4).to_numpy(),
})
table = table.sort_values("n", ascending=False).reset_index(drop=True)
table = table.head(50)

mlca_runtime.save_table(table, "target_counts.csv", caption="Target class counts and shares")

flagged = table.loc[table["share"] < 0.05, "class"].astype(str).tolist()
print(f"rows={total} classes={len(table)}")
print("flagged (<5%): " + (", ".join(flagged) if flagged else "none"))

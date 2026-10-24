import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = list(protocol.get("targets") or [])
if not targets:
    raise KeyError(f"No targets in _protocol. Keys: {sorted(protocol.keys())}")
target = targets[0]

df = pd.read_parquet(inputs["discovery"])
if target not in df.columns:
    raise KeyError(f"Target '{target}' not in discovery columns")

s = df[target].astype("string").str.strip()
ph = {"?", "-", "NA", "N/A", ""}
mask = s.isna() | s.isin(ph)
n_ph = int(mask.sum())
if n_ph:
    print(f"placeholder tokens found: {n_ph} in 1 column(s)")
s = s[~mask]

vc = s.value_counts(dropna=True)
total = int(vc.sum())
if total == 0:
    raise ValueError("No usable target values after cleaning")

out = pd.DataFrame({
    "class": [str(c) for c in vc.index],
    "n": vc.to_numpy().astype(int),
})
out["share"] = (out["n"] / total).round(4)
out = out.sort_values("n", ascending=False, ignore_index=True)

flagged = out.loc[out["share"] < 0.05, "class"].tolist()
mlca_runtime.save_table(out, "target_counts", "Target class counts and shares; classes <5% flagged")
mlca_runtime.record_metric("n_target_classes", float(len(out)), "train")
mlca_runtime.record_metric("n_classes_under_5pct", float(len(flagged)), "train")
print(f"rows={len(df)} classes={len(out)} flagged={len(flagged)}")

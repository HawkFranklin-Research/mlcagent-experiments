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
    raise KeyError(f"Target '{target}' not found in discovery data")

s = df[target].astype("string").str.strip()
placeholders = {"?", "-", "NA", "N/A", ""}
ph_mask = s.isna() | s.isin(placeholders)
n_ph = int(ph_mask.sum())
if n_ph:
    print(f"Placeholders found: {n_ph} in target column, converted to missing")
s = s[~ph_mask]

vc = s.value_counts(dropna=True)
total = int(vc.sum())
if total == 0:
    raise ValueError("No usable target values after cleaning")

rows = []
for cls, n in vc.items():
    n_int = int(n)
    rows.append({"class": str(cls), "n": n_int, "share": round(n_int / total, 4)})
out = pd.DataFrame(rows).sort_values("n", ascending=False, ignore_index=True)

flagged = out.loc[out["share"] < 0.05, "class"].tolist()

mlca_runtime.save_table(
    out,
    "target_counts.csv",
    "Target class counts and shares (flag <5%)",
)

mlca_runtime.record_metric("n_target_classes", float(len(out)), "train")
mlca_runtime.record_metric("n_classes_under_5pct", float(len(flagged)), "train")

print(f"rows={len(df)} classes={len(out)} flagged={len(flagged)}")

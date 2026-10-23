import json
import pandas as pd
import mlca_runtime as rt

rt.set_seed(42)
inputs = rt.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

if "targets" not in protocol:
    raise KeyError(f"Missing 'targets'. Available top-level keys: {sorted(protocol)}")
targets = protocol["targets"]
if not isinstance(targets, list) or not targets:
    raise ValueError("_protocol['targets'] must be a non-empty list")

df = pd.read_parquet(inputs["anemia"])
missing = [c for c in targets if c not in df.columns]
if missing:
    raise KeyError(f"Target columns missing from data: {missing}")

rows = []
total = len(df)
for t in targets:
    vc = df[t].value_counts(dropna=False)
    for cls, n in vc.items():
        label = "<missing>" if pd.isna(cls) else str(cls)
        rows.append({"target": t, "class": label, "n": int(n), "share": float(n / total) if total else float("nan")})

out = pd.DataFrame(rows)
out["share"] = out["share"].map(lambda x: float(f"{x:.4g}"))
if len(targets) == 1:
    out = out[["class", "n", "share"]]
else:
    out = out[["target", "class", "n", "share"]]

rt.save_table(out, "target_counts.csv", caption="Target class counts and shares")

for _, r in out.iterrows():
    rt.record_metric(f"count_{r['class']}", float(r["n"]), "all")
    rt.record_metric(f"share_{r['class']}", float(r["share"]), "all")

low = out[out["share"] < 0.05]
rt.record_metric("n_classes_under_5pct", float(len(low)), "all")

print(f"rows={total} targets={targets}")
print(f"classes={len(out)} under_5pct={len(low)}")
print("flagged=" + ("; ".join(f"{r['class']}:{r['share']}" for _, r in low.iterrows()) if len(low) else "none"))

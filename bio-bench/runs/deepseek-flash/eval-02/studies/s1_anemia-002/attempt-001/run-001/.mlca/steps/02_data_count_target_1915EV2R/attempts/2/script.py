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

total = len(df)
rows = []
for t in targets:
    for cls, n in df[t].value_counts(dropna=False).items():
        label = "<missing>" if pd.isna(cls) else str(cls)
        share = float(n / total) if total else float("nan")
        rows.append({"target": t, "class": label, "n": int(n), "share": round(share, 4)})

out = pd.DataFrame(rows)
if len(targets) == 1:
    out = out[["class", "n", "share"]]
else:
    out = out[["target", "class", "n", "share"]]

out = out.head(50)
rt.save_table(out, "target_counts.csv", caption="Target class counts and shares; classes under 5% are flagged")

low = out[out["share"] < 0.05]
print(f"rows={total} targets={targets}")
print(f"classes={len(out)} under_5pct={len(low)}")
print("flagged=" + ("; ".join(f"{r['class']}:{r['share']}" for _, r in low.iterrows()) if len(low) else "none"))

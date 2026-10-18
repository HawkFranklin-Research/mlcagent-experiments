import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

protocol = json.loads(paths["_protocol"].read_text())
try:
    targets = protocol["targets"]
except KeyError:
    raise KeyError(
        f"Missing 'targets' in _protocol. Available top-level keys: {sorted(protocol.keys())}"
    ) from None

if not isinstance(targets, (list, tuple)) or not targets:
    raise ValueError("_protocol['targets'] must be a non-empty list.")

targets = [str(t) for t in targets]
df = pd.read_parquet(paths["anuran_calls_uci"])

missing = [t for t in targets if t not in df.columns]
if missing:
    raise KeyError(f"Target columns missing from data: {missing}")

n_rows = len(df)
rows = []
for target in targets:
    counts = df[target].dropna().astype(str).value_counts()
    for cls, n in counts.items():
        rows.append({
            "class": f"{target}::{cls}",
            "n": int(n),
            "share": round(float(n) / n_rows, 4) if n_rows else float("nan"),
        })

out = pd.DataFrame(rows, columns=["class", "n", "share"])
mlca_runtime.save_table(out, "target_counts.csv", caption="Counts and shares of target classes.")

flagged = int((out["share"] < 0.05).sum())
mlca_runtime.record_metric("classes_under_5pct", flagged, "all")
mlca_runtime.record_metric("target_class_rows", len(out), "all")

print(f"rows={n_rows} targets={len(targets)}")
print(f"classes={len(out)} flagged_under_5pct={flagged}")
print("saved target_counts.csv")

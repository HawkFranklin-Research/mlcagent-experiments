import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise KeyError(f"Missing '{key}' in _protocol. Available keys: {list(proto.keys())}")

targets = proto["targets"]
if not targets:
    raise ValueError("No targets declared in _protocol['targets'].")

df = pd.read_parquet(inputs["discovery"])
target = targets[0]
if target not in df.columns:
    raise KeyError(f"Target '{target}' not in discovery columns: {list(df.columns)}")

n_total = len(df)
counts = df[target].value_counts(dropna=False)
share = counts / n_total

rows = []
for cls, n in counts.items():
    s = float(share.loc[cls])
    rows.append({
        "class": cls,
        "n": int(n),
        "share": round(s, 4),
        "flag_under_5pct": bool(s < 0.05),
    })
out = pd.DataFrame(rows, columns=["class", "n", "share", "flag_under_5pct"])

mlca_runtime.save_table(
    out, "target_counts.csv",
    caption=f"Class counts and shares for {target}; flag marks classes under 5% of rows.",
)

flagged = out.loc[out["flag_under_5pct"], "class"].astype(str).tolist()
print(f"rows={n_total}, classes={len(out)}, target={target}")
print(f"flagged (<5%): {flagged}")

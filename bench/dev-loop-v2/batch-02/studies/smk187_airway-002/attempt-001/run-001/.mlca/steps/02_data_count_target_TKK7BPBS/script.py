import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
if "targets" not in proto:
    raise KeyError(f"Missing 'targets'; available top-level keys: {list(proto)}")
targets = proto["targets"]
if not targets:
    raise ValueError("No targets declared in _protocol")

df = pd.read_parquet(inputs["smk187_airway_openml"])
if len(targets) != 1:
    raise ValueError(f"Expected exactly one target, got {len(targets)}")
tgt = targets[0]
if tgt not in df.columns:
    raise KeyError(f"Target column {tgt!r} not found in data")

vc = df[tgt].value_counts(dropna=False)
counts = vc.rename_axis("class").reset_index(name="n")
share = counts["n"] / len(df)
counts["share"] = share.map(lambda x: float(f"{x:.4g}"))
counts["under_5pct"] = share < 0.05

mlca_runtime.save_table(counts, "target_counts.csv", caption="Target class counts and shares")

n_classes = len(counts)
n_flag = int(counts["under_5pct"].sum())
print(f"target={tgt} rows={len(df)} classes={n_classes} flagged_under_5pct={n_flag}")
if n_flag:
    flagged = counts.loc[counts["under_5pct"], ["class", "share"]]
    print("flagged: " + ", ".join(f"{r['class']}={r['share']}" for _, r in flagged.iterrows()))
print("saved target_counts.csv")

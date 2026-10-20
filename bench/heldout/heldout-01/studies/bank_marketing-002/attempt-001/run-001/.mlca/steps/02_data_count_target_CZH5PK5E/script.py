import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")

df = pd.read_parquet(inputs["bank_marketing_uci"])
targets = protocol["targets"]
if not targets:
    raise ValueError(f"No targets declared; available keys: {list(protocol.keys())}")

rows_kept = []
for t in targets:
    if t not in df.columns:
        raise KeyError(f"Target '{t}' not in frozen table columns")
    vc = df[t].value_counts(dropna=False)
    total = int(vc.sum())
    out = pd.DataFrame({
        "class": vc.index.astype(str),
        "n": vc.values.astype(int),
        "share": (vc.values / total).astype(float),
    })
    out = out.sort_values("n", ascending=False).reset_index(drop=True)
    name = "target_counts.csv" if len(targets) == 1 else f"target_counts_{t}.csv"
    mlca_runtime.save_table(out, name, caption=f"Class counts for target '{t}'")
    flagged = out.loc[out["share"] < 0.05, "class"].tolist()
    mlca_runtime.record_metric(f"{t}_n_classes", float(len(out)), "train")
    mlca_runtime.record_metric(f"{t}_min_class_share", float(out["share"].min()), "train")
    mlca_runtime.record_metric(f"{t}_n_flagged_under_5pct", float(len(flagged)), "train")
    print(f"target={t} rows={total} classes={len(out)} min_share={out['share'].min():.4f}")
    print(f"flagged_under_5pct={flagged}")
    rows_kept.append(out)

if len(targets) == 1:
    print(f"saved target_counts.csv rows={len(rows_kept[0])}")

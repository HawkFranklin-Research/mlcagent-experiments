import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = proto["targets"]
if not targets:
    raise KeyError(f"_protocol['targets'] is empty; top-level keys: {sorted(proto.keys())}")
target = targets[0]

df = pd.read_parquet(inputs["wine_quality_uci"])
if target not in df.columns:
    raise KeyError(f"target '{target}' not in data; columns: {sorted(df.columns)}")

n_total = int(len(df))
vc = df[target].value_counts(dropna=False).sort_index()
tbl = pd.DataFrame({"class": vc.index, "n": vc.to_numpy(dtype=int)})
tbl["share"] = (tbl["n"] / n_total).round(4)
tbl = tbl.reset_index(drop=True)

flagged = tbl.loc[tbl["share"] < 0.05, "class"].tolist()

mlca_runtime.save_table(tbl, "target_counts.csv",
                        caption="Counts and shares per target class; flagged if share < 5%")
mlca_runtime.record_metric("n_classes", float(len(tbl)), split="train")
mlca_runtime.record_metric("n_classes_under_5pct", float(len(flagged)), split="train")

print(f"rows={n_total} classes={len(tbl)} flagged_under_5pct={len(flagged)}")
print(f"flagged classes: {flagged}")

import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
df = pd.read_parquet(inputs["wine_quality_uci"])

t = targets[0]
vc = df[t].value_counts().sort_index()
share = (vc / len(df)).round(4)
out = pd.DataFrame({"class": vc.index, "n": vc.to_numpy(), "share": share.to_numpy()})
under = (out["share"] < 0.05).to_numpy()

mlca_runtime.save_table(out, "target_counts.csv", "Target class counts and share of rows")

mlca_runtime.record_metric("n_rows", float(len(df)), "train")
mlca_runtime.record_metric("n_classes", float(len(vc)), "train")
mlca_runtime.record_metric("n_classes_under_5pct", float(int(under.sum())), "train")
mlca_runtime.record_metric("min_class_share", float(share.min()), "train")

print(f"rows={len(df)} classes={len(vc)} under5pct={out.loc[under, 'class'].tolist()}")

import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
targets = protocol["targets"]
target = targets[0]

df = pd.read_parquet(paths["wine_quality_uci"])
counts = df[target].value_counts(dropna=False).rename_axis("class").reset_index(name="n")
counts["share"] = counts["n"] / len(df)
counts = counts.sort_values("class").reset_index(drop=True)
counts["share"] = counts["share"].map(lambda v: float(f"{v:.4g}"))

threshold = 0.05
flagged = counts["share"] < threshold

mlca_runtime.save_table(counts, "target_counts.csv",
                        caption="Counts and shares per target class; classes under 5% flagged in metrics")

for _, row in counts.iterrows():
    mlca_runtime.record_metric(f"share_class_{row['class']}", float(row["share"]), "train")
mlca_runtime.record_metric("n_rows", int(len(df)), "train")
mlca_runtime.record_metric("n_classes", int(len(counts)), "train")
mlca_runtime.record_metric("n_flagged_classes", int(flagged.sum()), "train")

print(f"rows={len(df)} classes={len(counts)}")
print(f"flagged={int(flagged.sum())} classes under {threshold:.0%}")
print(f"flagged classes: {counts.loc[flagged, 'class'].tolist()}")

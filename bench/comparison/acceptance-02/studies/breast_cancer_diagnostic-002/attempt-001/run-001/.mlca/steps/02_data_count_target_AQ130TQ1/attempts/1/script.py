import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
if "targets" not in protocol:
    raise KeyError(f"Missing 'targets'; available keys: {sorted(protocol)}")

targets = protocol["targets"]
df = pd.read_parquet(paths["breast_cancer_diagnostic_uci"])
n_total = len(df)
rows = []
for target in targets:
    if target not in df.columns:
        raise KeyError(f"Target {target!r} not in data columns")
    counts = df[target].value_counts(dropna=False)
    for cls, n in counts.items():
        label = "NA" if pd.isna(cls) else str(cls)
        share = float(n) / float(n_total)
        rows.append({"class": label, "n": int(n), "share": share})
        mlca_runtime.record_metric(f"{target}_class_{label}_n", float(n), split="all")
        mlca_runtime.record_metric(f"{target}_class_{label}_share", share, split="all")

counts_table = pd.DataFrame(rows)
counts_table["share"] = counts_table["share"].map(lambda x: float(f"{x:.4g}"))
mlca_runtime.save_table(counts_table, "target_counts.csv", caption="Class counts and shares for target Diagnosis")

flagged = counts_table[counts_table["share"] < 0.05]["class"].tolist()
mlca_runtime.record_metric("classes_under_5pct", float(len(flagged)), split="all")
print(f"rows={n_total} targets={targets}")
print(f"classes={counts_table.to_dict('records')}")
print(f"flagged_under_5pct={flagged}")

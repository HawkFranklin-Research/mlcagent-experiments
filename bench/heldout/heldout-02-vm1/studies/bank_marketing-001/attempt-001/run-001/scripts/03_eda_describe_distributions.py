import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["bank_marketing_uci"])

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

# Build feature list: exclude targets, excluded, group, and key-like columns
feature_cols = []
for c in df.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = df[c]
    if (s.dtype == object or pd.api.types.is_integer_dtype(s.dtype)) and s.notna().all():
        if s.nunique(dropna=True) == len(s):
            continue
    feature_cols.append(c)

# Work copy: convert datetimes to numeric int64 for summarisation
work = df.copy()
for c in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(work[c]):
        work[c] = work[c].astype("int64")

def kind_of(s):
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return "numeric"
    return "categorical"

rows = []
target = targets[0]
classes = sorted(work[target].dropna().unique())

for grp_label, sub in [("overall", work)] + [(str(cls), work[work[target] == cls]) for cls in classes]:
    for c in feature_cols:
        s = sub[c]
        k = kind_of(s)
        row = {"variable": c, "kind": k, "group": grp_label}
        row["n_missing"] = int(s.isna().sum())
        if k == "numeric":
            row["mean"] = s.mean()
            row["sd"] = s.std()
            row["min"] = s.min()
            row["median"] = s.median()
            row["max"] = s.max()
        else:
            vc = s.value_counts(dropna=True)
            row["n_levels"] = int(vc.shape[0])
            row["top_share"] = float(vc.iloc[0] / vc.sum()) if vc.sum() > 0 else np.nan
        rows.append(row)

summary = pd.DataFrame(rows)
cols = ["variable", "kind", "group", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_share"]
summary = summary.reindex(columns=cols)

def round_sig(x, sig=4):
    if pd.isna(x):
        return np.nan
    if x == 0:
        return 0.0
    return float(f"{x:.{sig}g}")

for c in ["mean", "sd", "min", "median", "max", "n_levels", "top_share"]:
    summary[c] = summary[c].apply(lambda v: round_sig(v) if pd.notna(v) else np.nan)
summary["n_levels"] = summary["n_levels"].apply(lambda v: int(v) if pd.notna(v) else np.nan)

mlca_runtime.save_table(summary, "distributions.csv", caption="Distribution summary by feature and target class")

n_numeric = summary[summary["kind"] == "numeric"]["variable"].nunique()
n_categorical = summary[summary["kind"] == "categorical"]["variable"].nunique()
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"Features: {len(feature_cols)} (numeric={n_numeric}, categorical={n_categorical})")
print(f"Summary rows: {len(summary)}")

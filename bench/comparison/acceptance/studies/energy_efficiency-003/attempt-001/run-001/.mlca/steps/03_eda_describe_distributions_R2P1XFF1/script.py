import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

df = pd.read_parquet(paths["energy_efficiency_uci"])

# define feature columns
drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in df.columns if c not in drop]

# identify key-like columns
def is_keylike(s):
    if s.dtype.kind in "iu":
        return s.nunique(dropna=True) == len(s) and s.nunique() > 1
    if s.dtype == object or str(s.dtype).startswith("str"):
        try:
            return s.nunique(dropna=True) == len(s)
        except Exception:
            return False
    return False

keylike = [c for c in feature_cols if is_keylike(df[c])]
feature_cols = [c for c in feature_cols if c not in keylike]

# detect placeholder tokens
placeholders = {"?", "-", "NA", "N/A", ""}
n_placeholder_cols = 0
for c in df.columns:
    s = df[c]
    if s.dtype == object or str(s.dtype).startswith("str"):
        stripped = s.astype(str).str.strip()
        bad = stripped.isin(placeholders) & s.notna()
        if bad.any():
            n_placeholder_cols += 1

# summaries for all columns (features + targets, all present non-ignored)
summarize_cols = [c for c in df.columns if c not in set(excluded) and c not in keylike]
if group_col and group_col in summarize_cols:
    summarize_cols.remove(group_col)

rows = []
n_numeric = 0
n_categorical = 0
for c in summarize_cols:
    s = df[c]
    row = {"variable": c}
    if pd.api.types.is_numeric_dtype(s):
        row["kind"] = "numeric"
        row["mean"] = float(s.mean())
        row["sd"] = float(s.std())
        row["min"] = float(s.min())
        row["median"] = float(s.median())
        row["max"] = float(s.max())
        row["n_missing"] = int(s.isna().sum())
        n_numeric += 1
    else:
        row["kind"] = "categorical"
        vc = s.value_counts(dropna=True)
        row["n_levels"] = int(s.nunique(dropna=True))
        row["top_share"] = float(vc.iloc[0] / max(len(s), 1)) if len(vc) else np.nan
        row["n_missing"] = int(s.isna().sum())
        n_categorical += 1
    rows.append(row)

summary = pd.DataFrame(rows)
summary = summary.head(50)

# round numeric
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    if c in summary.columns:
        summary[c] = summary[c].apply(lambda v: round(float(v), 4) if pd.notna(v) else np.nan)

mlca_runtime.save_table(summary, "distributions.csv",
                        caption="Numeric and categorical column summaries for feature table and targets.")

mlca_runtime.record_fact_candidate("n_numeric", int(n_numeric), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(n_categorical), source="distributions.csv")

print(f"rows={len(df)} cols={len(df.columns)} features={len(feature_cols)}")
print(f"placeholder_cols={n_placeholder_cols}")
print(f"n_numeric={n_numeric} n_categorical={n_categorical}")

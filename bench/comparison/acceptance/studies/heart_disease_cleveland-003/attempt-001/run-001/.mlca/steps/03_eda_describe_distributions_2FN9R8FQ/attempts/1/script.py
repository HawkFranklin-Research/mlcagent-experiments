import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {sorted(protocol)}")

targets = protocol["targets"]
excluded = list(protocol["excluded_columns"])
task = protocol["task"]
group_column = protocol.get("group_column")

df = pd.read_parquet(paths["heart_disease_cleveland_uci"])

placeholders = {"?", "-", "NA", "N/A", ""}
placeholder_cols = []
for col in df.columns:
    if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
        original = df[col].astype("string")
        stripped = original.str.strip()
        mask = stripped.isin(placeholders) & original.notna()
        if mask.any():
            placeholder_cols.append(col)
            df[col] = stripped.mask(mask, np.nan)
        else:
            df[col] = stripped

excluded_set = set(excluded) | set(targets)
feature_cols = []
for col in df.columns:
    if col in excluded_set or col == group_column:
        continue
    lname = str(col).lower()
    if lname in {"id", "index", "row", "row_id", "key"} or lname.endswith("_id"):
        continue
    if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
        if df[col].notna().sum() == len(df) and df[col].nunique(dropna=True) == len(df):
            continue
    feature_cols.append(col)

rows = []
target_col = targets[0] if targets else None
groups = [("overall", df)]
if target_col in df.columns:
    for cls, sub in df.groupby(target_col, dropna=False):
        groups.append((str(cls), sub))

def numeric_row(sub, col, cls):
    s = sub[col]
    return {
        "variable": col,
        "type": "numeric",
        "target_class": cls,
        "mean": s.mean(),
        "sd": s.std(),
        "min": s.min(),
        "median": s.median(),
        "max": s.max(),
        "n_missing": int(s.isna().sum()),
        "n_levels": np.nan,
        "top_share": np.nan,
    }

def categorical_row(sub, col, cls):
    s = sub[col]
    vc = s.value_counts(dropna=True)
    n_nonmissing = int(s.notna().sum())
    top_share = float(vc.iloc[0] / n_nonmissing) if len(vc) and n_nonmissing else np.nan
    return {
        "variable": col,
        "type": "categorical",
        "target_class": cls,
        "mean": np.nan,
        "sd": np.nan,
        "min": np.nan,
        "median": np.nan,
        "max": np.nan,
        "n_missing": int(s.isna().sum()),
        "n_levels": int(s.nunique(dropna=True)),
        "top_share": top_share,
    }

for col in feature_cols:
    s = df[col]
    is_numeric = (
        pd.api.types.is_numeric_dtype(s)
        and not pd.api.types.is_bool_dtype(s)
        and s.nunique(dropna=True) > 10
    )
    for cls, sub in groups:
        rows.append(numeric_row(sub, col, cls) if is_numeric else categorical_row(sub, col, cls))

summary = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distribution summaries by target class")

n_numeric = int(summary.loc[summary["type"] == "numeric", "variable"].nunique())
n_categorical = int(summary.loc[summary["type"] == "categorical", "variable"].nunique())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"placeholders: {len(placeholder_cols)} columns" + (f" ({', '.join(map(str, placeholder_cols))})" if placeholder_cols else ""))
print(f"features: {n_numeric} numeric, {n_categorical} categorical")
print(f"summary rows: {len(summary)}")

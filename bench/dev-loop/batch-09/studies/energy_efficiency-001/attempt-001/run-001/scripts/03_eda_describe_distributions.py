import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["energy_efficiency_uci"])
proto = json.loads(inputs["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing '{k}' in _protocol. Available keys: {list(d.keys())}")
    return d[k]

targets = need(proto, "targets")
excluded = need(proto, "excluded_columns")
group_col = proto.get("group_column")
task = need(proto, "task")

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in df.columns if c not in drop]

def is_key_like(s):
    n = int(s.notna().sum())
    if n == 0 or int(s.nunique(dropna=True)) != n:
        return False
    if s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s):
        diffs = s.dropna().diff().dropna()
        return len(diffs) > 0 and diffs.nunique() == 1
    return False

feature_cols = [c for c in feature_cols if not is_key_like(df[c])]

placeholder_set = {"?", "-", "NA", "N/A", ""}
placeholder_cols = {}
for c in set(feature_cols + targets):
    if c not in df.columns:
        continue
    s = df[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        stripped = s.astype("string").str.strip()
        mask = stripped.isin(placeholder_set)
        if mask.any():
            placeholder_cols[c] = int(mask.sum())
            df[c] = stripped.mask(mask, pd.NA)
        else:
            df[c] = stripped

if placeholder_cols:
    print(f"Placeholders in {len(placeholder_cols)} columns: {', '.join(placeholder_cols)}")
else:
    print("No placeholder tokens found")

def summarize(s):
    vc = s.value_counts(dropna=True) if not pd.api.types.is_numeric_dtype(s) else None
    return {
        "n": int(s.notna().sum()),
        "n_missing": int(s.isna().sum()),
        "mean": s.mean() if pd.api.types.is_numeric_dtype(s) else np.nan,
        "sd": s.std(ddof=1) if pd.api.types.is_numeric_dtype(s) else np.nan,
        "min": s.min() if pd.api.types.is_numeric_dtype(s) else np.nan,
        "median": s.median() if pd.api.types.is_numeric_dtype(s) else np.nan,
        "max": s.max() if pd.api.types.is_numeric_dtype(s) else np.nan,
        "n_levels": int(vc.size) if vc is not None else np.nan,
        "top_share": float(vc.iloc[0] / vc.sum()) if vc is not None and vc.size else np.nan,
    }

rows = []
for group_name, cols in [("feature", feature_cols), ("target", targets)]:
    for c in cols:
        if c not in df.columns:
            continue
        d = summarize(df[c])
        d["variable"] = c
        d["type"] = "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical"
        d["group"] = group_name
        rows.append(d)

summary = pd.DataFrame(rows)
cols = ["variable", "type", "group", "n", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_share"]
summary = summary[cols]
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[c] = summary[c].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)
summary = summary.iloc[:50]

mlca_runtime.save_table(summary, "distributions.csv", caption="Column distribution summaries")

n_numeric = int((summary["type"] == "numeric").sum())
n_categorical = int((summary["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"Saved distributions.csv: {len(summary)} rows, {n_numeric} numeric, {n_categorical} categorical")

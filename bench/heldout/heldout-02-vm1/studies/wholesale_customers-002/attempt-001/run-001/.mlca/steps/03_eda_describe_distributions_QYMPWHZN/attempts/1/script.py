import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
required = ["task", "targets", "excluded_columns"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {sorted(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
df = pd.read_parquet(inputs["wholesale_customers_uci"])
drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)

def is_key_like(col, s):
    n = len(s)
    if n == 0 or s.nunique(dropna=False) != n:
        return False
    name = str(col).lower()
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s) and any(t in name for t in ("id", "code", "key", "index")):
        return True
    return False

features = [c for c in df.columns if c not in drop]
features = [c for c in features if not is_key_like(c, df[c])]

numeric_cols, cat_cols = [], []
for c in features:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        cat_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
            uniq = set(s.dropna().unique())
            if len(uniq) <= 10 and not uniq.issubset({0, 1}):
                cat_cols.append(c)
            else:
                numeric_cols.append(c)
        else:
            numeric_cols.append(c)
    else:
        cat_cols.append(c)

placeholders = ["?", "-", "NA", "N/A", ""]
placeholder_cols = 0
for c in cat_cols:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        stripped = s.astype("string").str.strip()
        is_ph = stripped.isin(placeholders)
        if is_ph.any():
            placeholder_cols += 1
        df[c] = stripped.mask(is_ph, pd.NA)

def rs(x):
    if pd.isna(x):
        return np.nan
    return float(f"{float(x):.4g}")

rows = []
for c in numeric_cols:
    s = df[c]
    rows.append({
        "feature": c, "type": "numeric", "n_missing": int(s.isna().sum()),
        "mean": rs(s.mean()), "sd": rs(s.std()), "min": rs(s.min()),
        "median": rs(s.median()), "max": rs(s.max()),
        "n_levels": np.nan, "top_level_share": np.nan,
    })
for c in cat_cols:
    s = df[c]
    vc = s.value_counts(normalize=True, dropna=True)
    rows.append({
        "feature": c, "type": "categorical", "n_missing": int(s.isna().sum()),
        "mean": np.nan, "sd": np.nan, "min": np.nan,
        "median": np.nan, "max": np.nan,
        "n_levels": int(s.nunique(dropna=True)),
        "top_level_share": rs(vc.iloc[0]) if len(vc) else np.nan,
    })

summary = pd.DataFrame(rows).head(50)
mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distribution summaries")
n_numeric = int((summary["type"] == "numeric").sum())
n_categorical = int((summary["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"features={len(features)} numeric={n_numeric} categorical={n_categorical}")
print(f"placeholders={placeholders} columns={placeholder_cols}")

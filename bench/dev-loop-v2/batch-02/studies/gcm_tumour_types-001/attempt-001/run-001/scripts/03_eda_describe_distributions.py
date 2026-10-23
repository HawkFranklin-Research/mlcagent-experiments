import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["gcm_tumour_types_openml"])
protocol = json.loads(inputs["_protocol"].read_text())

def req(obj, key):
    if key not in obj:
        raise KeyError(f"Missing protocol key '{key}'; available keys: {sorted(obj.keys())}")
    return obj[key]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
task = req(protocol, "task")
group_col = protocol.get("group_column")

drop_cols = set(targets) | set(excluded)
if group_col is not None:
    drop_cols.add(group_col)

def key_like(s):
    if len(s) <= 1 or s.nunique(dropna=True) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        vals = set(pd.unique(s.dropna()))
        return not vals.issubset({0, 1})
    return False

feature_cols = [c for c in df.columns if c not in drop_cols and not key_like(df[c])]
num_cols, cat_cols = [], []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    elif (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
          or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(s)):
        cat_cols.append(c)

placeholders = {"?", "-", "NA", "N/A"}
ph_cols = 0
for c in cat_cols:
    stripped = df[c].astype("string").str.strip()
    if stripped.isin(list(placeholders) | [""]).any():
        ph_cols += 1

def numeric_summary(sub, cols):
    if not cols:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    x = sub[cols]
    means = x.mean(numeric_only=True)
    sds = x.std(numeric_only=True)
    mins = x.min(numeric_only=True)
    meds = x.median(numeric_only=True)
    maxs = x.max(numeric_only=True)
    return (
        float(means.mean()), float(sds.mean()), float(mins.min()),
        float(meds.median()), float(maxs.max()), int(x.isna().sum().sum()),
    )

def categorical_summary(sub, cols):
    if not cols:
        return (np.nan, np.nan)
    levels, top_shares = [], []
    for c in cols:
        vc = sub[c].value_counts(dropna=True)
        levels.append(sub[c].nunique(dropna=True))
        top_shares.append(float(vc.iloc[0] / vc.sum()) if len(vc) else np.nan)
    return float(np.nanmedian(levels)), float(np.nanmean(top_shares))

def rs(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, np.integer)):
        return int(v)
    return float(f"{float(v):.4g}")

rows = []
def add_row(group, target_class, ftype, n_columns, num_stats, cat_stats):
    rows.append({
        "group": group, "target_class": target_class, "feature_type": ftype,
        "n_columns": n_columns,
        "mean": num_stats[0], "sd": num_stats[1], "min": num_stats[2],
        "median": num_stats[3], "max": num_stats[4], "n_missing": num_stats[5],
        "n_levels": cat_stats[0], "top_level_share": cat_stats[1],
    })

add_row("overall", "ALL", "numeric", len(num_cols), numeric_summary(df, num_cols), (np.nan, np.nan))
add_row("overall", "ALL", "categorical", len(cat_cols), (np.nan,) * 6, categorical_summary(df, cat_cols))

if len(targets) == 1:
    target_col = targets[0]
    for cls in sorted(df[target_col].dropna().unique(), key=str):
        sub = df[df[target_col] == cls]
        add_row("by_target_class", str(cls), "numeric", len(num_cols),
                numeric_summary(sub, num_cols), (np.nan, np.nan))
        if cat_cols:
            add_row("by_target_class", str(cls), "categorical", len(cat_cols),
                    (np.nan,) * 6, categorical_summary(sub, cat_cols))

dist = pd.DataFrame(rows)
for c in ["mean", "sd", "min", "median", "max", "n_levels", "top_level_share"]:
    dist[c] = dist[c].map(rs)
dist = dist[["group", "target_class", "feature_type", "n_columns", "mean", "sd",
             "min", "median", "max", "n_missing", "n_levels", "top_level_share"]]

mlca_runtime.save_table(dist, "distributions.csv",
                        caption="Feature distribution summary by feature type and target class.")
mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")

print(f"features: numeric={len(num_cols)}, categorical={len(cat_cols)}, total={len(feature_cols)}")
print(f"saved distributions.csv rows={len(dist)} cols={len(dist.columns)}")
print(f"placeholder tokens found in {ph_cols} text columns")

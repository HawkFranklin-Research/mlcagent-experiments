import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"] or [])
group_col = protocol["group_column"]
target = targets[0]

df = pd.read_parquet(inputs["gcm_tumour_types_openml"])
if target not in df.columns:
    raise KeyError(f"Target {target!r} not in data columns")

def is_key_like(s):
    try:
        if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
            return s.nunique(dropna=True) == len(s)
        if pd.api.types.is_integer_dtype(s.dtype):
            vals = s.dropna()
            if len(vals) != len(s) or vals.nunique() != len(vals):
                return False
            arr = np.sort(vals.to_numpy())
            return np.array_equal(arr, np.arange(len(arr)))
    except Exception:
        return False
    return False

exclude = set(targets) | set(excluded)
if group_col is not None:
    exclude.add(group_col)
feature_cols = [c for c in df.columns if c not in exclude and not is_key_like(df[c])]

numeric_cols = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
]
cat_cols = [
    c for c in feature_cols
    if c not in numeric_cols and not pd.api.types.is_datetime64_any_dtype(df[c])
]

def summarize_group(sub):
    res = {"n_numeric": len(numeric_cols), "n_categorical": len(cat_cols)}
    if numeric_cols:
        stats = sub[numeric_cols].agg(["mean", "std", "min", "median", "max", "count"])
        res["num_mean"] = stats.loc["mean"].mean()
        res["num_sd"] = stats.loc["std"].mean()
        res["num_min"] = stats.loc["min"].min()
        res["num_median"] = stats.loc["median"].median()
        res["num_max"] = stats.loc["max"].max()
        res["num_n_missing"] = int(sub[numeric_cols].isna().sum().sum())
    else:
        res.update({k: np.nan for k in ["num_mean", "num_sd", "num_min", "num_median", "num_max"]})
        res["num_n_missing"] = 0

    if cat_cols:
        levels, shares = [], []
        for c in cat_cols:
            s = sub[c].dropna()
            if len(s) == 0:
                continue
            levels.append(s.nunique())
            shares.append(s.value_counts(normalize=True).iloc[0])
        res["cat_n_levels_mean"] = float(np.mean(levels)) if levels else np.nan
        res["cat_top_share_mean"] = float(np.mean(shares)) if shares else np.nan
    else:
        res["cat_n_levels_mean"] = np.nan
        res["cat_top_share_mean"] = np.nan
    return res

rows = [{"group": "all", **summarize_group(df)}]
for cls in sorted(df[target].dropna().unique(), key=lambda x: str(x)):
    rows.append({"group": str(cls), **summarize_group(df[df[target] == cls])})

columns = [
    "group", "n_numeric", "n_categorical",
    "num_mean", "num_sd", "num_min", "num_median", "num_max", "num_n_missing",
    "cat_n_levels_mean", "cat_top_share_mean",
]
summary = pd.DataFrame(rows, columns=columns)

def sig4(v):
    try:
        if pd.isna(v):
            return v
        return 0.0 if v == 0 else float(f"{v:.4g}")
    except Exception:
        return v

for c in ["num_mean", "num_sd", "num_min", "num_median", "num_max", "cat_n_levels_mean", "cat_top_share_mean"]:
    summary[c] = summary[c].map(sig4)

mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distribution summaries by target class")
mlca_runtime.record_fact_candidate("n_numeric", int(len(numeric_cols)), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(len(cat_cols)), source="distributions.csv")

print(f"rows={len(df)} features={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(cat_cols)}")
print(f"saved distributions.csv rows={len(summary)} cols={len(summary.columns)}")

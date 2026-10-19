import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = proto["targets"]
excluded = proto["excluded_columns"]
group_col = proto.get("group_column")
df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in df.columns if c not in drop]


def is_datetime_col(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        v = s.dropna().astype(str)
        if len(v) == 0:
            return False
        try:
            pd.to_datetime(v.iloc[: min(50, len(v))])
            return True
        except Exception:
            return False
    return False


def r4(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return float(f"{x:.4g}")


rows = []
for c in feature_cols:
    s = df[c]
    n_missing = int(s.isna().sum())
    r = {"column": c, "n_missing": n_missing,
         "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
         "n_levels": np.nan, "top_share": np.nan}
    if is_datetime_col(s):
        r["kind"] = "datetime"
        r["n_levels"] = int(s.dropna().astype(str).nunique())
    elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        r["kind"] = "numeric"
        s_num = s.astype("float64")
        r["mean"] = r4(s_num.mean())
        r["sd"] = r4(s_num.std())
        r["min"] = r4(s_num.min())
        r["median"] = r4(s_num.median())
        r["max"] = r4(s_num.max())
        ns = s_num.dropna()
        r["n_levels"] = int(ns.nunique())
        vc = ns.value_counts(normalize=True)
        r["top_share"] = r4(vc.iloc[0]) if len(vc) else np.nan
    else:
        r["kind"] = "categorical"
        ns = s.astype("object").where(s.notna())
        vc = ns.value_counts(normalize=True, dropna=True)
        r["n_levels"] = int(ns.nunique())
        r["top_share"] = r4(vc.iloc[0]) if len(vc) else np.nan
    rows.append(r)

dist = pd.DataFrame(rows)[
    ["column", "kind", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_share"]
]
mlca_runtime.save_table(dist, "distributions.csv", caption="Numeric and categorical summaries")

n_num = int((dist["kind"] == "numeric").sum())
n_cat = int((dist["kind"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")

print(f"rows={dist.shape[0]} cols={dist.shape[1]}")
print(f"n_numeric={n_num} n_categorical={n_cat}")

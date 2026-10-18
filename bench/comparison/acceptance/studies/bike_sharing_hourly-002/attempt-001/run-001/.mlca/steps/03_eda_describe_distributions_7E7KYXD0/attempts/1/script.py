import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    group_col = protocol["group_column"]
    task = protocol["task"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available keys: {list(protocol.keys())}") from e

targets = [targets] if isinstance(targets, str) else list(targets)
excluded = [excluded] if isinstance(excluded, str) else list(excluded)
group_cols = [] if group_col is None else ([group_col] if isinstance(group_col, str) else list(group_col))
blocked = set(targets + excluded + group_cols)

df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])

def is_key_like(s: pd.Series) -> bool:
    nonmiss = s.dropna()
    if len(nonmiss) == 0 or nonmiss.nunique() != len(nonmiss):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        vals = np.sort(nonmiss.unique())
        return len(vals) > 1 and bool(np.all(np.diff(vals) == 1))
    return False

feature_cols = [c for c in df.columns if c not in blocked and not is_key_like(df[c])]

PH = {"?", "-", "NA", "N/A"}
ph_cols = []
found_tokens = set()
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        st = s.astype("string").str.strip()
        mask = st.isin(PH) | st.eq("")
        if mask.any():
            ph_cols.append(c)
            for v in st[mask].dropna().unique():
                found_tokens.add("<whitespace>" if v == "" else str(v))
print(f"placeholders={sorted(found_tokens) if found_tokens else 'none'} columns={len(ph_cols)}")

df_work = df.copy()
for c in feature_cols:
    if pd.api.types.is_object_dtype(df_work[c]) or pd.api.types.is_string_dtype(df_work[c]):
        st = df_work[c].astype("string").str.strip()
        df_work[c] = st.mask(st.isin(PH) | st.eq(""))

def is_categorical_col(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= 31:
        return True
    return False

rows = []
for c in feature_cols:
    s = df_work[c]
    n_missing = int(s.isna().sum())
    if is_categorical_col(s):
        non = s.dropna()
        n_levels = int(non.nunique())
        top_share = float(non.value_counts(normalize=True).iloc[0]) if len(non) else np.nan
        rows.append({
            "variable": c, "type": "categorical", "n_missing": n_missing,
            "n_levels": n_levels, "top_level_share": top_share,
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan, "target_class": "all",
        })
    else:
        x = pd.to_numeric(s, errors="coerce")
        n_missing = int(x.isna().sum())
        if x.notna().any():
            mean = float(x.mean())
            sd = float(x.std(ddof=1)) if x.count() > 1 else np.nan
            min_v = float(x.min())
            median = float(x.median())
            max_v = float(x.max())
        else:
            mean = sd = min_v = median = max_v = np.nan
        rows.append({
            "variable": c, "type": "numeric", "n_missing": n_missing,
            "n_levels": np.nan, "top_level_share": np.nan,
            "mean": mean, "sd": sd, "min": min_v,
            "median": median, "max": max_v, "target_class": "all",
        })

out = pd.DataFrame(rows, columns=[
    "variable", "type", "n_missing", "n_levels", "top_level_share",
    "mean", "sd", "min", "median", "max", "target_class",
])
for col in ["top_level_share", "mean", "sd", "min", "median", "max"]:
    out[col] = out[col].map(lambda v: np.nan if pd.isna(v) else float(f"{v:.4g}"))

n_numeric = int((out["type"] == "numeric").sum())
n_categorical = int((out["type"] == "categorical").sum())
mlca_runtime.save_table(out, "distributions.csv", caption="Feature distribution summaries.")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"features={len(feature_cols)} numeric={n_numeric} categorical={n_categorical}")
print(f"saved distributions.csv rows={len(out)} cols={out.shape[1]}")

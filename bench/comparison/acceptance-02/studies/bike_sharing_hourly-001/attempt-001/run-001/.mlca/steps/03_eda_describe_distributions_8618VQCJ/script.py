import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in proto]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {sorted(proto)}")
if not isinstance(proto["task"], dict) or "primary_metric" not in proto["task"]:
    raise KeyError("Missing nested protocol key task['primary_metric']")

targets = list(proto["targets"])
excluded = set(proto["excluded_columns"])
group = proto["group_column"]
task = proto["task"]

df = pd.read_parquet(paths["bike_sharing_hourly_uci"])

def is_keylike(s):
    n = len(s)
    if n == 0 or s.nunique(dropna=True) != n:
        return False
    if pd.api.types.is_float_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return False
    if pd.api.types.is_integer_dtype(s):
        name = str(s.name).lower()
        if any(tok in name for tok in ("id", "key", "index", "code")):
            return True
        vals = s.dropna().sort_values()
        return bool((vals.diff().dropna() == 1).all())
    return bool(
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )

protected = set(targets) | excluded | ({group} if group is not None else set())
feature_cols = []
for c in df.columns:
    if c in protected or is_keylike(df[c]):
        continue
    feature_cols.append(c)

text_cols = [
    c for c in feature_cols
    if pd.api.types.is_object_dtype(df[c])
    or pd.api.types.is_string_dtype(df[c])
    or isinstance(df[c].dtype, pd.CategoricalDtype)
]
placeholder_tokens = ["?", "-", "NA", "N/A"]
found = set()
placeholder_cols = 0
for c in text_cols:
    stripped = df[c].astype("string").str.strip()
    col_found = {tok for tok in placeholder_tokens if (stripped == tok).any()}
    if (stripped == "").any() and df[c].notna().any():
        col_found.add("whitespace")
    if col_found:
        placeholder_cols += 1
        found |= col_found

rows = []
for c in feature_cols:
    s = df[c]
    row = {"column": c, "n_missing": int(s.isna().sum())}
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        row.update({
            "type": "numeric",
            "mean": s.mean(),
            "sd": s.std(),
            "min": s.min(),
            "median": s.median(),
            "max": s.max(),
            "n_levels": np.nan,
            "top_share": np.nan,
        })
    else:
        vc = s.value_counts(dropna=True)
        total = int(vc.sum())
        row.update({
            "type": "categorical",
            "mean": np.nan,
            "sd": np.nan,
            "min": np.nan,
            "median": np.nan,
            "max": np.nan,
            "n_levels": int(vc.size),
            "top_share": float(vc.iloc[0] / total) if total else np.nan,
        })
    rows.append(row)

out = pd.DataFrame(rows, columns=[
    "column", "type", "n_missing", "mean", "sd", "min",
    "median", "max", "n_levels", "top_share",
])
for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[col] = out[col].map(
        lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan
    )

mlca_runtime.save_table(out, "distributions.csv", caption="Feature distribution summaries")
n_numeric = int((out["type"] == "numeric").sum())
n_categorical = int((out["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"features={len(feature_cols)} numeric={n_numeric} categorical={n_categorical}")
print(f"placeholders: {sorted(found) if found else 'none'}; columns={placeholder_cols}")
print(f"saved distributions.csv rows={len(out)}")

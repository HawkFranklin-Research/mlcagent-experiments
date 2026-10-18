import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["energy_efficiency_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

try:
    targets = list(protocol["targets"])
    excluded = list(protocol["excluded_columns"])
except KeyError as exc:
    raise KeyError(f"Missing protocol key {exc}; available keys: {list(protocol)}") from exc
group_col = protocol.get("group_column")

feature_cols = [c for c in df.columns if c not in set(targets) and c not in set(excluded) and c != group_col]

def is_key_like(s):
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        vals = s.dropna()
        return len(vals) > 0 and vals.nunique() == len(vals)
    if pd.api.types.is_integer_dtype(s):
        vals = s.dropna()
        return len(vals) > 0 and vals.nunique() == len(vals) and (s.is_monotonic_increasing or s.is_monotonic_decreasing)
    return False

feature_cols = [c for c in feature_cols if not is_key_like(df[c])]

placeholders = {"?", "-", "NA", "N/A"}
def clean_text(s):
    def one(v):
        if not isinstance(v, str):
            return v
        v2 = v.strip()
        return pd.NA if v2 == "" or v2 in placeholders else v2
    return s.map(one)

text_cols = [c for c in feature_cols if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
placeholder_cols = 0
for c in text_cols:
    vals = df[c].dropna()
    if any(isinstance(v, str) and (v.strip() in placeholders or v.strip() == "" or v != v.strip()) for v in vals):
        placeholder_cols += 1

rows = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s):
        rows.append({
            "column": c, "kind": "numeric", "n_missing": int(s.isna().sum()),
            "mean": s.mean(), "sd": s.std(ddof=1), "min": s.min(), "median": s.median(),
            "max": s.max(), "n_levels": np.nan, "top_level_share": np.nan,
        })
    else:
        cs = clean_text(s)
        vc = cs.value_counts(dropna=True)
        rows.append({
            "column": c, "kind": "categorical", "n_missing": int(cs.isna().sum()),
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
            "n_levels": int(cs.nunique(dropna=True)),
            "top_level_share": float(vc.iloc[0] / vc.sum()) if len(vc) else np.nan,
        })

summary = pd.DataFrame(rows, columns=[
    "column", "kind", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_level_share"
])
for c in ["mean", "sd", "min", "median", "max", "top_level_share"]:
    summary[c] = summary[c].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)

n_numeric = int((summary["kind"] == "numeric").sum())
n_categorical = int((summary["kind"] == "categorical").sum())

mlca_runtime.save_table(summary, "distributions.csv", caption="Column distribution summaries")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"summary rows={len(summary)} numeric={n_numeric} categorical={n_categorical}")
print(f"placeholder tokens: {placeholder_cols} columns")

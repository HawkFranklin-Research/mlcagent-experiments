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
    group = protocol["group_column"]
except KeyError as exc:
    raise KeyError(f"Missing protocol key {exc}; available keys: {sorted(protocol.keys())}") from exc

df = pd.read_parquet(inputs["seeds_uci"])
drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
features = [c for c in df.columns if c not in drop]

def is_key_like(s: pd.Series) -> bool:
    s = s.dropna()
    if s.empty:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique() == len(s)
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if set(s.unique()).issubset({0, 1}):
            return False
        return s.nunique() == len(s)
    return False

features = [c for c in features if not is_key_like(df[c])]

placeholders = {"?", "-", "NA", "N/A", ""}
placeholder_tokens = set()
placeholder_cols = 0
for c in features:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype("string").str.strip()
        mask = stripped.isin(placeholders)
        if mask.any():
            placeholder_cols += 1
            placeholder_tokens.update(stripped[mask].dropna().unique().tolist())
        df[c] = stripped.mask(mask, pd.NA)

print(f"Placeholders {sorted(placeholder_tokens) if placeholder_tokens else 'none'} in {placeholder_cols} text columns")

rows = []
for c in features[:50]:
    s = df[c]
    missing = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        rows.append({
            "feature": c, "type": "numeric",
            "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "median": s.median(), "max": s.max(), "n_missing": missing,
            "n_levels": np.nan, "top_level_share": np.nan,
        })
    else:
        s2 = s.dropna()
        n_levels = int(s2.nunique())
        top_share = float(s2.value_counts(normalize=True).iloc[0]) if len(s2) else np.nan
        rows.append({
            "feature": c, "type": "categorical",
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan, "n_missing": missing,
            "n_levels": n_levels, "top_level_share": top_share,
        })

summary = pd.DataFrame(rows, columns=[
    "feature", "type", "mean", "sd", "min", "median", "max",
    "n_missing", "n_levels", "top_level_share",
])
for col in ["mean", "sd", "min", "median", "max", "top_level_share"]:
    summary[col] = summary[col].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else x)

mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distributions summary")
n_numeric = int((summary["type"] == "numeric").sum())
n_categorical = int((summary["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"features={len(features)} saved={len(summary)} numeric={n_numeric} categorical={n_categorical}")
print(f"rows={len(df)} cols={len(df.columns)}")

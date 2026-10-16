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
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")

targets = protocol["targets"] or []
excluded = protocol["excluded_columns"] or []
group = protocol["group_column"]
df = pd.read_parquet(inputs["seeds_uci"])

placeholders = {"?", "-", "NA", "N/A", ""}
placeholder_cols = []
found_placeholders = set()
for c in df.select_dtypes(include=["object", "string"]).columns:
    s = df[c].astype("string")
    stripped = s.str.strip()
    mask = stripped.isin(placeholders) | stripped.str.fullmatch(r"\s*", na=False)
    if mask.any():
        placeholder_cols.append(c)
        found_placeholders.update(stripped[mask].dropna().unique().tolist())
    df[c] = stripped.mask(mask, pd.NA)
if placeholder_cols:
    print(f"Placeholders found: {sorted(found_placeholders)} in {len(placeholder_cols)} columns")

drop = set(targets) | set(excluded) | ({group} if group is not None else set())
features = []
for c in df.columns:
    if c in drop:
        continue
    s = df[c]
    if pd.api.types.is_integer_dtype(s) and s.notna().all() and s.nunique() == len(s):
        if (s.is_monotonic_increasing or s.is_monotonic_decreasing) and (s.max() - s.min() + 1 == len(s)):
            continue
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == len(s):
        continue
    features.append(c)

rows = []
for c in features:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s):
        rows.append({
            "variable": c, "type": "numeric",
            "mean": s.mean(), "sd": s.std(), "min": s.min(), "median": s.median(), "max": s.max(),
            "n_missing": int(s.isna().sum()),
            "n_levels": np.nan, "top_level_share": np.nan
        })
    else:
        vc = s.value_counts(dropna=True, normalize=True)
        rows.append({
            "variable": c, "type": "categorical",
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
            "n_missing": int(s.isna().sum()),
            "n_levels": int(s.nunique(dropna=True)),
            "top_level_share": float(vc.iloc[0]) if len(vc) else np.nan
        })

out = pd.DataFrame(rows)
if out.empty:
    out = pd.DataFrame(columns=["variable", "type", "mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_level_share"])
if len(out) > 50:
    out = out.head(50)
out = out.round(4)

mlca_runtime.save_table(out, "distributions.csv", caption="Feature distributions")
n_numeric = int((out["type"] == "numeric").sum())
n_categorical = int((out["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"Saved distributions.csv with {len(out)} features: {n_numeric} numeric, {n_categorical} categorical")

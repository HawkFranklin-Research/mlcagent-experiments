import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
df = pd.read_parquet(inp["energy_efficiency_uci"])

def required(d, key):
    if key not in d:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {sorted(d.keys())}")
    return d[key]

targets = list(required(protocol, "targets"))
excluded = list(required(protocol, "excluded_columns") or [])
group_col = required(protocol, "group_column")

def key_like(s):
    if s.isna().any() or s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        vals = s.sort_values()
        return bool((vals.diff().dropna() == 1).all())
    return False

feature_cols = [
    c for c in df.columns
    if c not in targets and c not in excluded and c != group_col and not key_like(df[c])
]

placeholders = {"?", "-", "NA", "N/A", ""}
found_tokens = set()
ph_cols = 0
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype("string").str.strip()
        mask = stripped.isin(list(placeholders)) & s.notna()
        if mask.any():
            ph_cols += 1
            found_tokens.update(stripped[mask].dropna().unique().tolist())
print(f"Placeholder tokens {sorted(found_tokens)} in {ph_cols} column(s)")

rows = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        rows.append({
            "variable": c, "type": "numeric", "n": int(x.notna().sum()),
            "n_missing": int(x.isna().sum()), "mean": x.mean(), "sd": x.std(),
            "min": x.min(), "median": x.median(), "max": x.max(),
            "n_levels": np.nan, "top_level": np.nan, "top_share": np.nan,
        })
    else:
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            x = s.astype("string").str.strip()
            x = x.mask(x.isin(list(placeholders)))
        else:
            x = s
        vc = x.value_counts(dropna=True)
        denom = x.notna().sum()
        rows.append({
            "variable": c, "type": "categorical", "n": int(denom),
            "n_missing": int(x.isna().sum()), "mean": np.nan, "sd": np.nan,
            "min": np.nan, "median": np.nan, "max": np.nan,
            "n_levels": int(x.nunique(dropna=True)),
            "top_level": vc.index[0] if len(vc) else np.nan,
            "top_share": float(vc.iloc[0] / denom) if len(vc) and denom else np.nan,
        })

out = pd.DataFrame(rows, columns=[
    "variable", "type", "n", "n_missing", "mean", "sd",
    "min", "median", "max", "n_levels", "top_level", "top_share",
]).head(50).copy()

def sig4(v):
    if isinstance(v, (int, np.integer)):
        return v
    if isinstance(v, (float, np.floating)) and np.isfinite(v):
        return float(f"{v:.4g}")
    return v

out = out.apply(lambda col: col.map(sig4))
mlca_runtime.save_table(out, "distributions.csv", caption="Feature distribution summary")

n_numeric = int((out["type"] == "numeric").sum())
n_categorical = int((out["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"Rows={len(df)}, features={len(feature_cols)}")
print(f"Saved distributions.csv rows={len(out)} numeric={n_numeric} categorical={n_categorical}")

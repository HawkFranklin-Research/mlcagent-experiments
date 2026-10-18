import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def req(d, key):
    if key not in d:
        raise KeyError(f"Missing required key {key!r}; available top-level keys: {sorted(d.keys())}")
    return d[key]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
group_col = req(protocol, "group_column")
task = req(protocol, "task")

df = pd.read_parquet(inputs["anuran_calls_uci"])
n = len(df)

feature_cols = [c for c in df.columns if c not in set(targets) and c not in set(excluded) and c != group_col]
key_like = []
for c in feature_cols:
    s = df[c]
    if s.nunique(dropna=False) == n and (
        pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_integer_dtype(s)
    ):
        key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

ph_tokens = set()
ph_cols = set()
text_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
for c in text_cols:
    s = df[c].astype("string").str.strip()
    for tok in ["?", "-", "NA", "N/A", ""]:
        if (s == tok).any():
            ph_tokens.add(tok if tok != "" else "<whitespace>")
            ph_cols.add(c)
    df[c] = s.replace(["?", "-", "NA", "N/A", ""], pd.NA)

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

rows = []
for c in numeric_cols:
    s = df[c]
    rows.append({
        "variable": c, "kind": "numeric", "group": "overall", "level": np.nan,
        "mean": s.mean(), "sd": s.std(), "min": s.min(), "median": s.median(), "max": s.max(),
        "n_missing": int(s.isna().sum()), "n_levels": np.nan, "top_share": np.nan,
    })

for c in categorical_cols:
    s = df[c]
    vc = s.value_counts(dropna=True)
    rows.append({
        "variable": c, "kind": "categorical", "group": "overall", "level": np.nan,
        "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
        "n_missing": int(s.isna().sum()), "n_levels": int(s.nunique(dropna=True)),
        "top_share": (vc.iloc[0] / vc.sum()) if len(vc) else np.nan,
    })

for target in targets:
    if target not in df.columns:
        raise KeyError(f"Target {target!r} not found in table columns: {list(df.columns)}")
    s = df[target]
    vc = s.value_counts(dropna=True)
    total = vc.sum()
    for level, count in vc.items():
        rows.append({
            "variable": target, "kind": "target_class", "group": target, "level": str(level),
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
            "n_missing": int(s.isna().sum()), "n_levels": int(s.nunique(dropna=True)),
            "top_share": (count / total) if total else np.nan,
        })

out = pd.DataFrame(rows)
keep_cols = ["variable", "kind", "group", "level", "mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_share"]
out = out[keep_cols].head(50).reset_index(drop=True)

def sig4(v):
    if pd.isna(v) or not isinstance(v, (int, float, np.number)):
        return v
    if v == 0:
        return 0.0
    return float(f"{v:.4g}")

for col in ["mean", "sd", "min", "median", "max", "n_levels", "top_share"]:
    out[col] = out[col].map(sig4)

mlca_runtime.save_table(out, "distributions.csv", caption="Column distributions summary")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")

ph_list = ", ".join(sorted(ph_tokens)) if ph_tokens else "none"
print(f"Features: {len(feature_cols)} ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
print(f"Placeholders: {ph_list}; {len(ph_cols)} column(s) contain them")
print(f"Saved distributions.csv with {len(out)} rows")

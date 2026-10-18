import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
df = pd.read_parquet(inp["abalone_uci"])

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

def is_text(s):
    return pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object

def is_keylike(s):
    if is_text(s):
        n = int(s.notna().sum())
        return n > 0 and int(s.nunique(dropna=True)) == n
    if pd.api.types.is_integer_dtype(s) and s.name is not None:
        name = str(s.name).lower()
        if any(k in name for k in ("id", "key", "code")):
            n = int(s.notna().sum())
            return n > 0 and int(s.nunique(dropna=True)) == n
    return False

feature_cols = [
    c for c in df.columns
    if c not in targets and c not in excluded and c != group_col and not is_keylike(df[c])
]
summary_cols = [c for c in df.columns if c in feature_cols or c in targets]

placeholder_tokens = {"?", "-", "NA", "N/A", "na", "n/a", "nan", "NaN"}
found_tokens = set()
placeholder_cols = 0
clean = df.copy()
for c in df.columns:
    if is_text(df[c]):
        s = df[c].astype("string").str.strip()
        mask = s.isin(placeholder_tokens) | s.eq("")
        if mask.any():
            found_tokens.update(set(s[mask].dropna().unique().tolist()))
            placeholder_cols += 1
        clean[c] = s.mask(mask, pd.NA)

def round_sig(x, sig=4):
    if pd.isna(x):
        return np.nan
    if x == 0:
        return 0.0
    return float(f"{x:.{sig}g}")

rows = []
for c in summary_cols:
    s = clean[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        rows.append({
            "variable": c, "type": "numeric", "n_missing": int(v.isna().sum()),
            "mean": v.mean(), "sd": v.std(), "min": v.min(), "median": v.median(),
            "max": v.max(), "n_levels": np.nan, "top_share": np.nan,
        })
    else:
        s2 = s.astype("string").where(s.notna(), pd.NA)
        vals = s2.value_counts(dropna=True)
        n_ok = int(s2.notna().sum())
        top_share = float(vals.iloc[0] / n_ok) if len(vals) and n_ok else np.nan
        rows.append({
            "variable": c, "type": "categorical", "n_missing": int(s2.isna().sum()),
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan,
            "max": np.nan, "n_levels": float(len(vals)) if len(vals) else np.nan,
            "top_share": top_share,
        })

out = pd.DataFrame(rows)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[c] = out[c].map(round_sig)
out = out.sort_values(["type", "variable"]).reset_index(drop=True)

mlca_runtime.save_table(out, "distributions.csv", caption="Distribution summaries")
n_numeric = int((out["type"] == "numeric").sum())
n_categorical = int((out["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

tokens = ", ".join(sorted(found_tokens)) if found_tokens else "none"
print(f"placeholders found: {tokens}; columns containing placeholders: {placeholder_cols}")
print(f"summary rows={len(out)} numeric={n_numeric} categorical={n_categorical}")

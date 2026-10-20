import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def req(key):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")
    return protocol[key]

targets = list(req("targets"))
_ = list(req("excluded_columns"))
df = pd.read_parquet(inputs["breast_cancer_diagnostic_uci"])

df_clean = df.copy()
placeholder_cols = 0
found_tokens = set()
for c in df_clean.columns:
    if pd.api.types.is_object_dtype(df_clean[c]) or pd.api.types.is_string_dtype(df_clean[c]):
        s = df_clean[c].astype("string").str.strip()
        mask = s.str.upper().isin(["?", "-", "NA", "N/A", ""])
        if mask.any():
            placeholder_cols += 1
            found_tokens.update(s[mask].dropna().unique().tolist())
            s = s.mask(mask, pd.NA)
        df_clean[c] = s

print(f"placeholders: {sorted(found_tokens) if found_tokens else 'none'}; columns={placeholder_cols}")

rows = []
for c in df_clean.columns:
    s = df_clean[c]
    is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if is_num:
        s_num = pd.to_numeric(s, errors="coerce")
        rows.append({
            "variable": c, "type": "numeric", "group": "all",
            "n": int(s_num.notna().sum()), "n_missing": int(s_num.isna().sum()),
            "mean": s_num.mean(), "sd": s_num.std(), "min": s_num.min(),
            "median": s_num.median(), "max": s_num.max(),
            "n_levels": np.nan, "top_level_share": np.nan,
        })
    else:
        s_cat = s.astype("string")
        n = int(s_cat.notna().sum())
        vc = s_cat.value_counts(dropna=True)
        top_share = float(vc.iloc[0] / n) if n and len(vc) else np.nan
        rows.append({
            "variable": c, "type": "categorical", "group": "all",
            "n": n, "n_missing": int(s_cat.isna().sum()),
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_levels": int(vc.size), "top_level_share": top_share,
        })
        if c in targets:
            for lev, cnt in vc.items():
                rows.append({
                    "variable": c, "type": "categorical", "group": str(lev),
                    "n": int(cnt), "n_missing": np.nan,
                    "mean": np.nan, "sd": np.nan, "min": np.nan,
                    "median": np.nan, "max": np.nan,
                    "n_levels": np.nan,
                    "top_level_share": float(cnt / n) if n else np.nan,
                })

summary = pd.DataFrame(rows).iloc[:50].copy()
stat_cols = ["n", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_level_share"]
summary[stat_cols] = summary[stat_cols].apply(pd.to_numeric, errors="coerce").round(4)

uniq = summary[["variable", "type"]].drop_duplicates()
n_numeric = int((uniq["type"] == "numeric").sum())
n_categorical = int((uniq["type"] == "categorical").sum())

mlca_runtime.save_table(summary, "distributions.csv", caption="Column distribution summaries")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"numeric={n_numeric}, categorical={n_categorical}, rows={len(summary)}")

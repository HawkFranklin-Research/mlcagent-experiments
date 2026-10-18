import json
import pandas as pd
import numpy as np
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["adult_census_uci"])
protocol = json.loads(paths["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

key_like = []
for c in df.columns:
    if df[c].dtype == object or str(df[c].dtype).startswith("string"):
        if df[c].nunique(dropna=True) == df[c].notna().sum() and df[c].notna().sum() > 0:
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(df[c]):
        if df[c].nunique(dropna=True) == df[c].notna().sum() and df[c].notna().sum() == len(df):
            key_like.append(c)

drop_set = set(targets) | set(excluded) | ({group_col} if group_col else set()) | set(key_like)
feat_cols = [c for c in df.columns if c not in drop_set]

numeric_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in feat_cols if c not in numeric_cols]

rows = []
def add_row(target_val, cols_n, cols_c):
    for c in cols_n:
        s = pd.to_numeric(df[c], errors="coerce") if target_val is None else pd.to_numeric(df.loc[df[targets[0]] == target_val, c], errors="coerce")
        rows.append({"target_class": target_val if target_val is not None else "ALL",
                     "variable": c, "type": "numeric", "n": int(s.notna().sum()),
                     "mean": s.mean(), "sd": s.std(), "min": s.min(),
                     "median": s.median(), "max": s.max(),
                     "n_missing": int(s.isna().sum()),
                     "n_levels": np.nan, "top_share": np.nan})
    for c in cols_c:
        s = df[c] if target_val is None else df.loc[df[targets[0]] == target_val, c]
        vc = s.astype("string").value_counts(dropna=True)
        rows.append({"target_class": target_val if target_val is not None else "ALL",
                     "variable": c, "type": "categorical", "n": int(s.notna().sum()),
                     "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan,
                     "max": np.nan, "n_missing": int(s.isna().sum()),
                     "n_levels": int(vc.size),
                     "top_share": float(vc.iloc[0] / vc.sum()) if vc.size else np.nan})

add_row(None, numeric_cols, cat_cols)
tgt = targets[0]
for val in pd.unique(df[tgt].dropna()):
    add_row(val, numeric_cols, cat_cols)

out = pd.DataFrame(rows)
num_stats = ["mean", "sd", "min", "median", "max", "top_share"]
out[num_stats] = out[num_stats].round(4)
if len(out) > 50:
    out = out.head(50)
mlca_runtime.save_table(out, "distributions.csv", caption="Column distribution summary by target class")

mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")
print(f"rows={len(df)} features={len(feat_cols)} numeric={len(numeric_cols)} categorical={len(cat_cols)}")
print(f"summary_rows={len(out)} key_like_dropped={key_like}")

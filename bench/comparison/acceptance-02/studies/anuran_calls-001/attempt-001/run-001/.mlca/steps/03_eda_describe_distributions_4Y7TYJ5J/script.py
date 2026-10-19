import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "task"]:
    if k not in protocol:
        raise KeyError(f"protocol missing {k!r}; top-level keys: {list(protocol.keys())}")
df = pd.read_parquet(inputs["anuran_calls_uci"])

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []) or [])
group_col = protocol.get("group_column")

# Feature columns (RL-0007)
feature_cols = []
for c in df.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = df[c]
    try:
        if s.nunique(dropna=False) == len(df) and not pd.api.types.is_float_dtype(s):
            continue
    except Exception:
        pass
    feature_cols.append(c)

num_cols, cat_cols = [], []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        cat_cols.append(c)

# Placeholder check (RL-0023)
placeholders = ["?", "-", "NA", "N/A", ""]
ph_cols = 0
for c in cat_cols:
    s = df[c].astype(str).str.strip().mask(df[c].isna(), np.nan)
    if s.isin(placeholders).any():
        ph_cols += 1
print(f"placeholder tokens in {ph_cols} column(s)")

rows = []
for c in num_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    rows.append(dict(variable=c, dtype=str(df[c].dtype), role="numeric",
                     n_missing=int(s.isna().sum()),
                     mean=float(s.mean()) if s.notna().any() else np.nan,
                     sd=float(s.std()) if s.notna().sum() > 1 else np.nan,
                     min=float(s.min()) if s.notna().any() else np.nan,
                     median=float(s.median()) if s.notna().any() else np.nan,
                     max=float(s.max()) if s.notna().any() else np.nan,
                     n_levels=np.nan, top_share=np.nan,
                     n_rows=int(s.notna().sum())))

for c in cat_cols:
    s = df[c].astype(str).str.strip().mask(df[c].isna(), np.nan)
    s = s.replace("", np.nan)
    vc = s.value_counts(dropna=True)
    rows.append(dict(variable=c, dtype=str(df[c].dtype), role="categorical",
                     n_missing=int(s.isna().sum()),
                     mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                     n_levels=int(vc.shape[0]),
                     top_share=float(vc.iloc[0] / vc.sum()) if len(vc) else np.nan,
                     n_rows=int(s.notna().sum())))

for t in targets:
    if t not in df.columns:
        continue
    s = df[t].astype(str).str.strip().mask(df[t].isna(), np.nan)
    vc = s.value_counts(dropna=True)
    tot = int(vc.sum())
    n_lv = int(vc.shape[0])
    if n_lv <= 12:
        for lvl, cnt in vc.items():
            rows.append(dict(variable=f"{t}:{lvl}", dtype=str(df[t].dtype),
                             role="target_class", n_missing=np.nan,
                             mean=np.nan, sd=np.nan, min=np.nan, median=np.nan,
                             max=np.nan, n_levels=n_lv,
                             top_share=float(cnt / tot) if tot else np.nan,
                             n_rows=int(cnt)))
    else:
        rows.append(dict(variable=t, dtype=str(df[t].dtype), role="target",
                         n_missing=int(s.isna().sum()),
                         mean=np.nan, sd=np.nan, min=np.nan, median=np.nan,
                         max=np.nan, n_levels=n_lv,
                         top_share=float(vc.iloc[0] / tot) if tot else np.nan,
                         n_rows=tot))

dist = pd.DataFrame(rows).head(50)
for c in ["mean", "sd", "min", "median", "max", "n_levels", "top_share"]:
    dist[c] = pd.to_numeric(dist[c], errors="coerce")

mlca_runtime.save_table(dist, "distributions.csv")
mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")
print(f"features: numeric={len(num_cols)} categorical={len(cat_cols)}")
print(f"distributions rows={len(dist)}")

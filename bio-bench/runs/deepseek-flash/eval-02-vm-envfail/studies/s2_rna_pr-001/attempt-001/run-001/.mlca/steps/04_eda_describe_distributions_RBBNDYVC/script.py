import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
df = pd.read_parquet(inp["rna_pr"])
proto = json.loads(inp["_protocol"].read_text())

targets = list(proto["targets"])
excluded = set(proto["excluded_columns"])
group_col = proto.get("group_column") if isinstance(proto, dict) else None
target = targets[0]

feat_cols = [c for c in df.columns
             if c not in set(targets) and c not in excluded and c != group_col]

def is_keylike(s):
    if pd.api.types.is_string_dtype(s) or s.dtype.kind in "OUS":
        nn = int(s.notna().sum())
        return nn > 0 and s.nunique(dropna=True) == nn
    return False

numeric_cols, cat_cols = [], []
for c in feat_cols:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
        continue
    if pd.api.types.is_numeric_dtype(s):
        if s.dtype.kind in "iu" and s.notna().sum() > 0 and s.nunique(dropna=True) == s.notna().sum():
            vals = s.dropna().astype(float)
            if vals.max() - vals.min() + 1 == len(vals):
                continue
        numeric_cols.append(c)
    elif is_keylike(s):
        continue
    else:
        if s.nunique(dropna=True) <= max(20, int(0.05 * len(df))):
            cat_cols.append(c)

def num_summary(cols, frame):
    if not cols:
        return dict(n_numeric_cols=0, num_mean_of_means=np.nan, num_mean_of_sd=np.nan,
                    num_min=np.nan, num_mean_of_median=np.nan, num_max=np.nan,
                    num_mean_missing_frac=np.nan)
    means, sds, mins, meds, maxs, miss = [], [], [], [], [], []
    for c in cols:
        s = frame[c]
        means.append(s.mean()); sds.append(s.std()); mins.append(s.min())
        meds.append(s.median()); maxs.append(s.max()); miss.append(s.isna().mean())
    return dict(n_numeric_cols=len(cols),
                num_mean_of_means=float(np.nanmean(means)),
                num_mean_of_sd=float(np.nanmean(sds)),
                num_min=float(np.nanmin(mins)),
                num_mean_of_median=float(np.nanmean(meds)),
                num_max=float(np.nanmax(maxs)),
                num_mean_missing_frac=float(np.nanmean(miss)))

def cat_summary(cols, frame):
    if not cols:
        return dict(n_categorical_cols=0, cat_mean_n_levels=np.nan, cat_mean_top_share=np.nan)
    nlev, top = [], []
    for c in cols:
        s = frame[c].dropna()
        if len(s) == 0:
            continue
        nlev.append(s.nunique())
        top.append(s.value_counts(normalize=True).iloc[0])
    return dict(n_categorical_cols=len(cols),
                cat_mean_n_levels=float(np.mean(nlev)) if nlev else np.nan,
                cat_mean_top_share=float(np.mean(top)) if top else np.nan)

scopes = [("all", df)]
if target in df.columns:
    for v in df[target].dropna().unique():
        scopes.append((f"class_{v}", df[df[target] == v]))

rows = []
for name, sub in scopes:
    r = {"scope": name, "n_rows": int(len(sub))}
    r.update(num_summary(numeric_cols, sub))
    r.update(cat_summary(cat_cols, sub))
    rows.append(r)

dist = pd.DataFrame(rows)

def sig4(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return x
    try:
        return float(f"{float(x):.4g}")
    except Exception:
        return x

for col in dist.columns:
    if dist[col].dtype.kind in "fc":
        dist[col] = dist[col].map(sig4)

mlca_runtime.save_table(dist, "distributions.csv",
                        caption="Column distributions by target class (aggregate statistics).")
mlca_runtime.record_fact_candidate("n_numeric", int(len(numeric_cols)), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(len(cat_cols)), source="distributions.csv")

print(f"features={len(feat_cols)} numeric={len(numeric_cols)} categorical={len(cat_cols)}")
print(f"rows={len(df)} scopes={len(scopes)}")

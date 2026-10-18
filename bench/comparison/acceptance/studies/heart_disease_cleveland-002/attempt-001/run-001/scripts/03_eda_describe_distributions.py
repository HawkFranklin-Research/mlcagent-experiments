import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column", None)
df = pd.read_parquet(inputs["heart_disease_cleveland_uci"])

drop_set = set(targets) | set(excluded)
if group_col:
    drop_set.add(group_col)
candidates = [c for c in df.columns if c not in drop_set]

def looks_integral(s):
    v = s.dropna()
    if len(v) == 0:
        return False
    try:
        return bool((v % 1 == 0).all())
    except Exception:
        return False

feat_cols = []
for c in candidates:
    s = df[c]
    n = len(s)
    nun = s.nunique(dropna=True)
    if s.dtype == object or str(s.dtype) == 'category':
        if nun == n:
            continue
    elif looks_integral(s) and nun == n and n > 1:
        continue
    feat_cols.append(c)

numeric_cols, categorical_cols = [], []
for c in feat_cols:
    s = df[c]
    if s.dtype == object or str(s.dtype) == 'category' or pd.api.types.is_bool_dtype(s):
        categorical_cols.append(c)
    elif looks_integral(s) and s.nunique(dropna=True) <= 10:
        categorical_cols.append(c)
    else:
        numeric_cols.append(c)

def sig4(x):
    if pd.isna(x) or x == 0:
        return x
    return float(f"{float(x):.4g}")

def summarize(sub, label):
    rows = []
    for c in numeric_cols:
        s = sub[c]
        rows.append(dict(group=label, variable=c, kind="numeric",
                         mean=sig4(s.mean()), sd=sig4(s.std()), min=sig4(s.min()),
                         median=sig4(s.median()), max=sig4(s.max()),
                         n_levels=np.nan, top_level_share=np.nan,
                         n_missing=int(s.isna().sum())))
    for c in categorical_cols:
        s = sub[c]
        vc = s.value_counts(dropna=True)
        top = float(vc.iloc[0] / vc.sum()) if len(vc) else np.nan
        rows.append(dict(group=label, variable=c, kind="categorical",
                         mean=np.nan, sd=np.nan, min=np.nan, median=np.nan,
                         max=np.nan, n_levels=int(s.nunique(dropna=True)),
                         top_level_share=sig4(top),
                         n_missing=int(s.isna().sum())))
    return rows

rows = summarize(df, "all")
t = targets[0]
for cls in sorted(pd.Series(df[t].dropna().unique()).tolist()):
    rows.extend(summarize(df[df[t] == cls], f"{t}={cls}"))

out = pd.DataFrame(rows)[
    ["group","variable","kind","mean","sd","min","median","max",
     "n_levels","top_level_share","n_missing"]
].head(50)
mlca_runtime.save_table(out, "distributions.csv",
                        caption="Feature distributions overall and by target class")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")
print(f"rows={len(df)} numeric={len(numeric_cols)} categorical={len(categorical_cols)}")
print(f"summary_rows={len(out)}")

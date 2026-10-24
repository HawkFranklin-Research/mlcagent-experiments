import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
missing = [k for k in ("targets", "excluded_columns", "task") if k not in proto]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available keys: {list(proto)}")
targets = proto["targets"]
excluded = list(proto.get("excluded_columns", []))
group_col = proto.get("group_column")

df = pd.read_parquet(inp["wine_quality_uci"])
n = len(df)

blocked = set(targets) | set(excluded) | ({group_col} if group_col else set())
features = [c for c in df.columns if c not in blocked]
# key-like: object/code column whose values are all distinct and carry no measurement
features = [c for c in features
            if not (df[c].dtype.kind in "OUS" and df[c].nunique(dropna=True) == n)]

num_cols, cat_cols = [], []
for c in features:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if s.dtype.kind in "ifu" or s.dtype.kind == "b":
        if s.dtype.kind == "i" and s.nunique(dropna=True) <= 1:
            continue
        num_cols.append(c)
    else:
        cat_cols.append(c)
print(f"rows={n} numeric={len(num_cols)} categorical={len(cat_cols)}")


def r4(x):
    return x if pd.isna(x) else float(f"{x:.4g}")


def cat_stats(s):
    vc = s.dropna().value_counts()
    tot = int(vc.sum())
    return int(vc.size), (float(vc.iloc[0] / tot) if tot else np.nan), tot


rows = []
for c in num_cols:
    s = df[c]
    rows.append(dict(group="all", column=c, type="numeric", n=int(s.notna().sum()),
                     mean=r4(s.mean()), sd=r4(s.std()), min=r4(s.min()),
                     median=r4(s.median()), max=r4(s.max()),
                     n_missing=int(s.isna().sum()), n_levels=np.nan, top_share=np.nan))
for c in cat_cols:
    s = df[c]
    nl, ts, tot = cat_stats(s)
    rows.append(dict(group="all", column=c, type="categorical", n=tot,
                     mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                     n_missing=int(s.isna().sum()), n_levels=nl, top_share=r4(ts)))

T = targets[0]
if cat_cols and df[T].nunique(dropna=True) <= 7:
    for cls, sub in df.groupby(T, dropna=True):
        for c in cat_cols:
            nl, ts, tot = cat_stats(sub[c])
            rows.append(dict(group=f"{T}={cls}", column=c, type="categorical", n=tot,
                             mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                             n_missing=int(sub[c].isna().sum()), n_levels=nl, top_share=r4(ts)))

dist = pd.DataFrame(rows).head(50)
cols = ["group", "column", "type", "n", "mean", "sd", "min", "median", "max",
        "n_missing", "n_levels", "top_share"]
dist = dist[cols]
mlca_runtime.save_table(dist, "distributions.csv",
                        "Per-column distribution summaries; overall and per target class.")
mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")
print(f"groups={dist['group'].nunique()} rows={len(dist)}")

import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in proto:
        raise KeyError(f"missing protocol key {k!r}; available: {list(proto.keys())}")

targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group_col = proto.get("group_column")
task = proto["task"]

df = pd.read_parquet(inputs["heart_disease_cleveland_uci"])
reserved = set(targets) | set(excluded)
if group_col:
    reserved.add(group_col)
cand = [c for c in df.columns if c not in reserved]


def is_keylike(s):
    n = s.notna().sum()
    if n < 2:
        return False
    if s.nunique(dropna=True) != n:
        return False
    if s.dtype == object or str(s.dtype).startswith("string"):
        return True
    if pd.api.types.is_integer_dtype(s):
        return True
    return False


features = [c for c in cand if not is_keylike(df[c])]
thresh = max(2, int(0.05 * len(df)))

numeric_cols, cat_cols, dt_cols = [], [], []
for c in features:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif s.nunique(dropna=True) <= thresh:
        cat_cols.append(c)

for c in dt_cols:
    d = pd.to_datetime(df[c], errors="coerce")
    for part in ("month", "dayofweek", "hour"):
        newc = f"{c}_{part}"
        df[newc] = getattr(d.dt, part)
        numeric_cols.append(newc)

target = targets[0]
classes = sorted(df[target].dropna().unique().tolist())


def num_stats(s, mask):
    ss = s[mask]
    if ss.notna().sum() == 0:
        return dict(mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan, n_missing=int(mask.sum()))
    return dict(mean=round(float(ss.mean()), 4), sd=round(float(ss.std()), 4) if ss.notna().sum() > 1 else np.nan,
                min=round(float(ss.min()), 4), median=round(float(ss.median()), 4),
                max=round(float(ss.max()), 4), n_missing=int(ss.isna().sum()))


def cat_stats(s, mask):
    ss = s[mask].astype("object")
    n = ss.notna().sum()
    if n == 0:
        return dict(n_missing=int(ss.isna().sum()), n_levels=0, top_share=np.nan)
    return dict(n_missing=int(ss.isna().sum()), n_levels=int(ss.nunique()),
                top_share=round(float(ss.value_counts().iloc[0]) / n, 4))


rows = []
base_cols = ["variable", "kind", "group", "mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_share"]
all_mask = pd.Series(True, index=df.index)
for c in numeric_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    r = dict(variable=c, kind="numeric", group="all", n_levels=np.nan, top_share=np.nan)
    r.update(num_stats(s, all_mask)); rows.append(r)
    for cls in classes:
        r = dict(variable=c, kind="numeric", group=str(cls), n_levels=np.nan, top_share=np.nan)
        r.update(num_stats(s, df[target] == cls)); rows.append(r)
for c in cat_cols:
    s = df[c]
    r = dict(variable=c, kind="categorical", group="all", mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan)
    r.update(cat_stats(s, all_mask)); rows.append(r)
    for cls in classes:
        r = dict(variable=c, kind="categorical", group=str(cls), mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan)
        r.update(cat_stats(s, df[target] == cls)); rows.append(r)

out = pd.DataFrame(rows)[base_cols].head(50)
mlca_runtime.save_table(out, "distributions.csv",
                        caption="Numeric and categorical feature summaries, overall and per target class")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")
print(f"n_numeric={len(numeric_cols)} n_categorical={len(cat_cols)} rows={len(out)}")

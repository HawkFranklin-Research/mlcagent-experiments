import json
import numpy as np
import pandas as pd
import mlca_runtime
from pandas.api import types as ptypes

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
avail = sorted(protocol.keys())
for k in ("targets", "excluded_columns", "group_column"):
    if k not in protocol:
        raise ValueError(f"_protocol missing '{k}'; available keys: {avail}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"] or [])
group_col = protocol["group_column"]
target = targets[0]

df = pd.read_parquet(inputs["smk187_airway_openml"])
feature_cols = [c for c in df.columns
                if c not in targets and c not in excluded and c != group_col]

work = df
dt_cols = [c for c in feature_cols if ptypes.is_datetime64_any_dtype(df[c])]
if dt_cols:
    work = df.copy()
    for c in dt_cols:
        work[c] = (work[c] - work[c].min()).dt.total_seconds()

num_cols = [c for c in feature_cols if ptypes.is_numeric_dtype(work[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

placeholders = {"?", "-", "NA", "N/A"}
hit_cols = 0
for c in cat_cols:
    vals = set(df[c].astype("string").str.strip().dropna().unique().tolist())
    if vals & placeholders:
        hit_cols += 1
print(f"placeholder columns: {hit_cols}")

classes = sorted(pd.unique(df[target].dropna()).tolist())
rows = []

def add(variable, kind, group, **kw):
    row = dict(variable=variable, kind=kind, group=group)
    row.update(kw)
    rows.append(row)

nan_stats = dict(mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                 n_levels=np.nan, top_share=np.nan)

def num_block(sub, group):
    agg = sub[num_cols].agg(["mean", "std", "min", "median", "max"])
    miss = sub[num_cols].isna().sum()
    for c in num_cols:
        add(c, "numeric", group, mean=agg.at["mean", c], sd=agg.at["std", c],
            min=agg.at["min", c], median=agg.at["median", c], max=agg.at["max", c],
            n_missing=int(miss[c]), n_levels=np.nan, top_share=np.nan)

if num_cols:
    num_block(work, "all")
    for cls in classes:
        num_block(work[work[target] == cls], f"target={cls}")

def cat_stats(s):
    n = s.notna().sum()
    vc = s.value_counts(dropna=True)
    nl = int(vc.shape[0])
    ts = float(vc.iloc[0] / n) if nl and n else np.nan
    return nl, ts, int(s.isna().sum())

for c in cat_cols:
    nl, ts, mi = cat_stats(df[c])
    add(c, "categorical", "all", n_levels=nl, top_share=ts, n_missing=mi, **nan_stats)
    for cls in classes:
        nl, ts, mi = cat_stats(df.loc[df[target] == cls, c])
        add(c, "categorical", f"target={cls}", n_levels=nl, top_share=ts,
            n_missing=mi, **nan_stats)

out = pd.DataFrame(rows, columns=["variable", "kind", "group", "mean", "sd", "min",
                                  "median", "max", "n_missing", "n_levels", "top_share"])
for col in ("mean", "sd", "min", "median", "max", "top_share"):
    out[col] = out[col].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)

mlca_runtime.save_table(out, "distributions.csv",
                        caption="Per-column distributions overall and by target class")

n_num = int(out.loc[out.kind == "numeric", "variable"].nunique())
n_cat = int(out.loc[out.kind == "categorical", "variable"].nunique())
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")
print(f"rows={len(out)} numeric={n_num} categorical={n_cat} classes={len(classes)}")

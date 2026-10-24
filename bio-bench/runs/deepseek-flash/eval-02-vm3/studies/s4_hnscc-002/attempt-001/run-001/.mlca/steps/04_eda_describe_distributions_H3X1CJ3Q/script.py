import mlca_runtime, json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["discovery"])

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")

def is_keylike(s):
    try:
        if s.isna().any():
            return False
        if pd.api.types.is_numeric_dtype(s):
            return False
        if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.StringDtype):
            if s.nunique() == len(s) and s.astype(str).str.len().max() <= 12:
                return True
    except Exception:
        pass
    return False

skip = set(targets) | set(excluded)
if group_col:
    skip.add(group_col)

num_cols, cat_cols = [], []
for c in df.columns:
    if c in skip:
        continue
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        df[c + "_month"] = pd.to_datetime(s, errors="coerce").dt.month
        cat_cols.append(c + "_month")
        continue
    if is_keylike(s):
        continue
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        cat_cols.append(c)

placeholder_cols = {}
for c in cat_cols:
    if c not in df.columns:
        continue
    s = df[c].astype(str).str.strip()
    hits = [ph for ph in ["?", "-", "NA", "N/A"] if (s == ph).any()]
    if hits:
        placeholder_cols[c] = hits

def num_stats(s):
    return dict(n=len(s), n_missing=int(s.isna().sum()),
                mean=float(s.mean()), sd=float(s.std()),
                min=float(s.min()), median=float(s.median()),
                max=float(s.max()), n_levels=np.nan, top_share=np.nan)

def cat_stats(s):
    vc = s.value_counts(dropna=True)
    return dict(n=len(s), n_missing=int(s.isna().sum()),
                mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                n_levels=int(s.nunique(dropna=True)),
                top_share=float(vc.iloc[0] / vc.sum()) if len(vc) else np.nan)

def summarize(sub, grp):
    return [{"group": grp, "variable": c, "dtype": "numeric", **num_stats(sub[c])} for c in num_cols] + \
           [{"group": grp, "variable": c, "dtype": "categorical", **cat_stats(sub[c])} for c in cat_cols]

rows = summarize(df, "all")
tgt = targets[0]
if tgt in df.columns:
    for cls, sub in df.groupby(tgt, dropna=True):
        rows += summarize(sub, f"{tgt}={cls}")

summary = pd.DataFrame(rows)
kept = []
for grp in pd.unique(summary["group"]):
    sub = summary[summary["group"] == grp]
    for dt in ("numeric", "categorical"):
        ss = sub[sub["dtype"] == dt].sort_values("n_missing", ascending=False).head(8)
        if len(ss):
            kept.append(ss)
cols = ["group", "variable", "dtype", "n", "n_missing",
        "mean", "sd", "min", "median", "max", "n_levels", "top_share"]
summary = pd.concat(kept).head(50).reset_index(drop=True)[cols]
for c in ("mean", "sd", "min", "median", "max", "top_share"):
    summary[c] = summary[c].astype(float).round(4)

mlca_runtime.save_table(summary, "distributions.csv",
                        caption="Distribution summary of numeric and categorical columns")
mlca_runtime.record_fact_candidate("n_numeric", int(len(num_cols)),
                                   unit="count", source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(len(cat_cols)),
                                   unit="count", source="distributions.csv")

print(f"rows={len(df)} cols={df.shape[1]}")
print(f"n_numeric={len(num_cols)} n_categorical={len(cat_cols)}")
print(f"placeholder_tokens in {len(placeholder_cols)} text columns")
print(f"target_classes={df[tgt].value_counts(dropna=True).to_dict() if tgt in df.columns else 'n/a'}")

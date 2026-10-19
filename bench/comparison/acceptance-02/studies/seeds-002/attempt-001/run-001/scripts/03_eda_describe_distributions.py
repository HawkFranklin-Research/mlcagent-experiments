import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["seeds_uci"])

def get_key(d, k, required=False, default=None):
    if k in d:
        return d[k]
    if required:
        raise KeyError(f"missing '{k}'; available keys: {list(d.keys())}")
    return default

targets = get_key(protocol, "targets", required=True)
excluded = get_key(protocol, "excluded_columns", required=True)
group_col = get_key(protocol, "group_column", default=None)

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feat_cols = [c for c in df.columns if c not in drop]

def is_keylike(s):
    s2 = s.dropna()
    if len(s2) == 0 or len(s2) != len(s):
        return False
    if s2.nunique() != len(s2):
        return False
    if pd.api.types.is_float_dtype(s):
        return False  # continuous measurement, never an identifier
    return True

feat_cols = [c for c in feat_cols if not is_keylike(df[c])]
num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in feat_cols if c not in num_cols]

rows = []
for c in num_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    rows.append(dict(column=c, type="numeric", n_missing=int(s.isna().sum()),
                     mean=s.mean(), sd=s.std(), min=s.min(), median=s.median(),
                     max=s.max(), n_levels=np.nan, top_share=np.nan))
for c in cat_cols:
    s = df[c].astype("object")
    vc = s.dropna().value_counts()
    nl = int(vc.shape[0])
    tot = int(vc.sum())
    rows.append(dict(column=c, type="categorical", n_missing=int(s.isna().sum()),
                     mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                     n_levels=nl, top_share=(float(vc.iloc[0]) / tot) if tot else np.nan))

dist = pd.DataFrame(rows)[["column", "type", "n_missing", "mean", "sd", "min",
                           "median", "max", "n_levels", "top_share"]]
mlca_runtime.save_table(dist, "distributions.csv", caption="Column distribution summary")

mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")

print(f"rows={len(df)} features={len(feat_cols)}")
print(f"numeric={len(num_cols)} categorical={len(cat_cols)}")

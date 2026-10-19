import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["anuran_calls_uci"])

missing = [k for k in ("targets", "excluded_columns", "task") if k not in proto]
if missing:
    raise KeyError(f"Missing protocol keys: {missing}. Available top-level keys: {list(proto.keys())}")

targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group_col = proto.get("group_column")

def is_keylike(s):
    if pd.api.types.is_integer_dtype(s) or s.dtype == object:
        return s.nunique(dropna=True) == len(s)
    return False

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]
feature_cols = [c for c in feature_cols if not is_keylike(df[c])]
numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in feature_cols if c not in numeric_cols]

ph_tokens = {"?", "-", "na", "n/a", ""}
found = []
for c in cat_cols:
    s = df[c].astype("string")
    n_ph = int(s.str.strip().str.lower().isin(ph_tokens).sum())
    if n_ph:
        found.append((c, n_ph))
print(f"placeholders found: {found if found else 'none'}")

rows = []
for c in numeric_cols:
    s = df[c]
    rows.append(dict(column=c, kind="numeric", mean=s.mean(), sd=s.std(),
                     min=s.min(), median=s.median(), max=s.max(),
                     n_missing=int(s.isna().sum()), n_levels=np.nan,
                     top_share=np.nan, target=np.nan, klass=np.nan, share=np.nan))
for c in cat_cols:
    s = df[c].astype("object")
    vc = s.value_counts(dropna=True)
    rows.append(dict(column=c, kind="categorical", mean=np.nan, sd=np.nan,
                     min=np.nan, median=np.nan, max=np.nan,
                     n_missing=int(s.isna().sum()), n_levels=int(vc.size),
                     top_share=float(vc.iloc[0] / len(s)) if len(vc) else np.nan,
                     target=np.nan, klass=np.nan, share=np.nan))

n_rows = len(df)
for t in targets:
    if t not in df.columns:
        continue
    vc = df[t].astype("object").value_counts(dropna=True)
    for cls, cnt in vc.items():
        rows.append(dict(column=t, kind="target_share", mean=np.nan, sd=np.nan,
                         min=np.nan, median=np.nan, max=np.nan, n_missing=np.nan,
                         n_levels=np.nan, top_share=np.nan, target=t,
                         klass=str(cls), share=float(cnt / n_rows)))

out = pd.DataFrame(rows)
if len(out) > 50:
    out = out.iloc[:50]
out = out.round(4)
mlca_runtime.save_table(out, "distributions.csv", caption="Feature and target distributions")

mlca_runtime.record_fact_candidate("n_numeric", int(len(numeric_cols)), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(len(cat_cols)), source="distributions.csv")

print(f"numeric={len(numeric_cols)} categorical={len(cat_cols)} summary_rows={len(out)}")

import mlca_runtime, json, pandas as pd, numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in proto:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(proto.keys())}")

targets = proto["targets"]
excluded = set(proto["excluded_columns"])
group_col = proto["group_column"]
df = pd.read_parquet(inputs["wine_quality_uci"])

feature_cols = []
for c in df.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = df[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == len(df):
        continue
    feature_cols.append(c)

rows = []
for c in feature_cols:
    s = df[c]
    n_missing = int(s.isna().sum())
    n_nonmiss = int(s.notna().sum())
    if pd.api.types.is_datetime64_any_dtype(s):
        s2 = (s - s.min()).dt.days.astype(float)
        rows.append(dict(variable=c, type="numeric", n=n_nonmiss, n_missing=n_missing,
                         mean=s2.mean(), sd=s2.std(), min=s2.min(), median=s2.median(), max=s2.max(),
                         n_levels=np.nan, top_share=np.nan, top_level=np.nan))
    elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        rows.append(dict(variable=c, type="numeric", n=n_nonmiss, n_missing=n_missing,
                         mean=s.mean(), sd=s.std(), min=s.min(), median=s.median(), max=s.max(),
                         n_levels=np.nan, top_share=np.nan, top_level=np.nan))
    else:
        vc = s.value_counts(dropna=True)
        n_levels = int(s.nunique(dropna=True))
        top_share = float(vc.iloc[0] / n_nonmiss) if n_nonmiss and len(vc) else np.nan
        top_level = vc.index[0] if len(vc) else np.nan
        rows.append(dict(variable=c, type="categorical", n=n_nonmiss, n_missing=n_missing,
                         mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                         n_levels=n_levels, top_share=top_share, top_level=top_level))

if targets:
    t = targets[0]
    if t in df.columns:
        vc = df[t].value_counts(dropna=False)
        for cls, cnt in vc.items():
            rows.append(dict(variable=t, type="target_class", n=int(cnt), n_missing=0,
                             mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                             n_levels=np.nan, top_share=float(cnt / len(df)), top_level=cls))

summary = pd.DataFrame(rows)
cols = ["variable", "type", "n", "n_missing", "mean", "sd", "min", "median",
        "max", "n_levels", "top_share", "top_level"]
summary = summary[cols]
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[c] = pd.to_numeric(summary[c], errors="coerce").round(4)

mlca_runtime.save_table(summary, "distributions.csv",
                        caption="Numeric and categorical feature summaries")

n_numeric = int((summary["type"] == "numeric").sum())
n_categorical = int((summary["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"rows={len(df)} features={len(feature_cols)} numeric={n_numeric} categorical={n_categorical}")

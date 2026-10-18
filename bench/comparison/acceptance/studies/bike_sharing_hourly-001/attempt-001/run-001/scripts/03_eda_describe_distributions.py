import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise KeyError(f"missing protocol key {k!r}; available top-level keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
task = protocol["task"]  # read but distributions do not depend on task type

df = pd.read_parquet(inp["bike_sharing_hourly_uci"])
n = len(df)

protected = set(targets) | set(excluded) | ({group} if group else set())
key_cols = []
for c in df.columns:
    if c in protected:
        continue
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if n > 1 and s.nunique(dropna=True) == n:
            key_cols.append(c)
    elif pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if n > 1 and s.nunique(dropna=True) == n:
            key_cols.append(c)

feature_cols = [c for c in df.columns if c not in protected and c not in key_cols]

placeholders = {"?", "-", "NA", "N/A", ""}
text_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
df_sum = df.copy()
found = {}
n_cols_with = 0
for c in text_cols:
    s = df_sum[c].astype("string").str.strip()
    mask = s.isin(placeholders)
    cnt = int(mask.sum())
    if cnt:
        n_cols_with += 1
        for tok in placeholders:
            k = int((s == tok).sum())
            if k:
                found[tok] = found.get(tok, 0) + k
    df_sum[c] = s.mask(mask, pd.NA)
print(f"text placeholders: {found} in {n_cols_with} columns")

rows = []
for c in feature_cols:
    s = df_sum[c]
    n_missing = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        rows.append(dict(variable=c, type="numeric", n=n, n_missing=n_missing,
                         mean=s.mean(), sd=s.std(), min=s.min(), median=s.median(), max=s.max(),
                         n_levels=np.nan, top_share=np.nan))
    else:
        vc = s.value_counts(dropna=True)
        top = float(vc.iloc[0] / vc.sum()) if len(vc) and vc.sum() > 0 else np.nan
        rows.append(dict(variable=c, type="categorical", n=n, n_missing=n_missing,
                         mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                         n_levels=int(s.nunique(dropna=True)), top_share=top))

dist = pd.DataFrame(rows)
cols = ["variable", "type", "n", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_share"]
dist = dist[cols]
for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    dist[col] = dist[col].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)

mlca_runtime.save_table(dist, "distributions.csv", caption="Numeric and categorical column distributions")
n_num = int((dist["type"] == "numeric").sum())
n_cat = int((dist["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")

print(f"features={len(feature_cols)} numeric={n_num} categorical={n_cat}")
print(f"saved distributions.csv rows={len(dist)} cols={dist.shape[1]}")

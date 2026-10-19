import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
if "targets" not in proto or "excluded_columns" not in proto:
    raise KeyError(f"Protocol missing targets/excluded_columns. Available keys: {list(proto.keys())}")
targets = proto["targets"]
excluded = proto["excluded_columns"]
group_col = proto.get("group_column")
df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])

# Placeholder token check for object/string columns
placeholder_tokens = ["?", "-", "NA", "N/A"]
cols_with_placeholders = []
for c in df.select_dtypes(include=["object", "string"]).columns:
    s = df[c]
    n_ph = s.isin(placeholder_tokens).sum() + s.astype(str).str.strip().eq("").sum()
    if n_ph > 0:
        cols_with_placeholders.append(c)
print(f"Placeholder tokens found in {len(cols_with_placeholders)} columns: {cols_with_placeholders[:3]}")

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]
# Drop key-like columns: object or integer with all distinct values
for c in list(feature_cols):
    s = df[c]
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype):
        if s.nunique(dropna=False) == len(df):
            feature_cols.remove(c)

rows = []
for c in feature_cols + targets:
    s = df[c]
    n_missing = int(s.isna().sum())
    is_num = pd.api.types.is_numeric_dtype(s.dtype) and not (
        pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=True) <= 25
    )
    if is_num:
        rows.append(dict(column=c, kind="numeric",
                         group="target" if c in targets else "feature",
                         n_missing=n_missing, mean=s.mean(), sd=s.std(),
                         min=s.min(), median=s.median(), max=s.max(),
                         n_levels=np.nan, top_share=np.nan))
    else:
        vc = s.value_counts(normalize=True, dropna=True)
        rows.append(dict(column=c, kind="categorical",
                         group="target" if c in targets else "feature",
                         n_missing=n_missing, mean=np.nan, sd=np.nan,
                         min=np.nan, median=np.nan, max=np.nan,
                         n_levels=int(s.nunique(dropna=True)),
                         top_share=float(vc.iloc[0]) if len(vc) > 0 else np.nan))

df_sum = pd.DataFrame(rows)
def round_sig(x, sig=4):
    if pd.isna(x) or not isinstance(x, (int, float, np.floating, np.integer)):
        return x
    return float(f"{x:.{sig}g}")
for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    df_sum[col] = df_sum[col].apply(round_sig)

mlca_runtime.save_table(df_sum, "distributions.csv", caption="Distributions summary")

n_numeric = int(((df_sum["group"] == "feature") & (df_sum["kind"] == "numeric")).sum())
n_categorical = int(((df_sum["group"] == "feature") & (df_sum["kind"] == "categorical")).sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"rows={len(df)}, features={len(feature_cols)}")
print(f"n_numeric={n_numeric}, n_categorical={n_categorical}")
print(f"summary shape={df_sum.shape}")

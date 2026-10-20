import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["steel_plates_faults_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column", None)

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group]

placeholder_cols = 0
tokens = {"?", "-", "NA", "N/A", ""}
for c in df.select_dtypes(include=["object", "string", "category"]).columns:
    s = df[c].astype("string").str.strip()
    mask = s.isin(tokens) & s.notna()
    if mask.any():
        placeholder_cols += 1

def is_key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return int(s.nunique(dropna=True)) == int(s.notna().sum()) and s.notna().sum() > 0
    return False

feature_cols = [c for c in feature_cols if not is_key_like(df[c])]
num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

rows = []
for c in num_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    rows.append(dict(column=c, kind="numeric", n=int(s.notna().sum()),
                     mean=s.mean(), sd=s.std(), min=s.min(), median=s.median(),
                     max=s.max(), n_missing=int(s.isna().sum()),
                     n_levels=np.nan, top_share=np.nan))
for c in cat_cols:
    s = df[c]
    vc = s.value_counts(dropna=True)
    top = float(vc.iloc[0] / s.notna().sum()) if len(vc) and s.notna().any() else np.nan
    rows.append(dict(column=c, kind="categorical", n=int(s.notna().sum()),
                     mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                     n_missing=int(s.isna().sum()), n_levels=int(s.nunique(dropna=True)),
                     top_share=top))
for t in targets:
    s = df[t]
    vc = s.value_counts(dropna=True)
    top = float(vc.iloc[0] / s.notna().sum()) if len(vc) and s.notna().any() else np.nan
    rows.append(dict(column=t, kind="target", n=int(s.notna().sum()),
                     mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                     n_missing=int(s.isna().sum()), n_levels=int(s.nunique(dropna=True)),
                     top_share=top))

out = pd.DataFrame(rows).head(50)
for c in out.select_dtypes(include=[np.number]).columns:
    out[c] = out[c].round(4)

mlca_runtime.save_table(out, "distributions.csv", caption="Feature and target distributions")
mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")

print(f"placeholders found in {placeholder_cols} text columns")
print(f"rows={len(out)} numeric={len(num_cols)} categorical={len(cat_cols)}")

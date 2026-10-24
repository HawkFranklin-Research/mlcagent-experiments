import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"_protocol missing '{key}'. Available top-level keys: {sorted(protocol)}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")

df = pd.read_parquet(inputs["wholesale_customers_uci"])

blocked = set(targets) | set(excluded) | ({group_col} if group_col else set())
feature_cols = []
for c in [c for c in df.columns if c not in blocked]:
    s = df[c]
    key_like = (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_integer_dtype(s)
    )
    if key_like and s.notna().sum() == len(s) and s.nunique(dropna=True) == len(s):
        continue
    feature_cols.append(c)

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols and not pd.api.types.is_datetime64_any_dtype(df[c])]

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols, ph_tokens = 0, set()
for c in categorical_cols:
    s = df[c].dropna().astype(str)
    hit = s.str.strip().isin(placeholders) | (s != s.str.strip())
    if hit.any():
        ph_cols += 1
        ph_tokens |= set(s[hit].str.strip().unique())

rows = []
for c in numeric_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    rows.append(dict(column=c, type="numeric", n_missing=int(s.isna().sum()),
                     mean=s.mean(), sd=s.std(ddof=1), min=s.min(), median=s.median(), max=s.max(),
                     n_levels=np.nan, top_share=np.nan))
for c in categorical_cols:
    s = df[c]
    vc = s.value_counts(normalize=True, dropna=True)
    rows.append(dict(column=c, type="categorical", n_missing=int(s.isna().sum()),
                     mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                     n_levels=int(s.nunique(dropna=True)),
                     top_share=float(vc.iloc[0]) if len(vc) else np.nan))

cols = ["column", "type", "mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_share"]
summary = pd.DataFrame(rows, columns=cols).head(50)
for c in ("mean", "sd", "min", "median", "max", "top_share"):
    summary[c] = summary[c].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)

mlca_runtime.save_table(summary, "distributions.csv", caption="Per-feature distribution summary (numeric and categorical).")

n_numeric = int((summary["type"] == "numeric").sum())
n_categorical = int((summary["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"frozen table {df.shape[0]} rows x {df.shape[1]} cols; {len(feature_cols)} feature cols")
print(f"numeric={n_numeric} categorical={n_categorical} (summarized in distributions.csv)")
print(f"placeholder tokens in {ph_cols} text columns: {sorted(ph_tokens)}")

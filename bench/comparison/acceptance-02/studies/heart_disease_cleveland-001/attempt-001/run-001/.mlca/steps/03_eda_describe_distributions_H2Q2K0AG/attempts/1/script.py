import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["heart_disease_cleveland_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

placeholder_cols = 0
placeholders = {"?", "-", "NA", "N/A", ""}
for col in df.columns:
    if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
        s = df[col].astype("string").str.strip()
        mask = s.isin(placeholders)
        if mask.any():
            placeholder_cols += 1
        df[col] = s.mask(mask, pd.NA)

feature_cols = []
for col in df.columns:
    if col in targets or col in excluded or col == group_col:
        continue
    s = df[col]
    nn = s.notna().sum()
    if nn > 1 and s.nunique(dropna=True) == nn:
        if (pd.api.types.is_integer_dtype(s) or
            pd.api.types.is_object_dtype(s) or
            pd.api.types.is_string_dtype(s) or
            isinstance(s.dtype, pd.CategoricalDtype)):
            continue
    feature_cols.append(col)

numeric_cols = [c for c in feature_cols
                if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
cat_cols = [c for c in feature_cols if c not in numeric_cols]

if len(targets) == 1 and targets[0] in df.columns:
    tgt = targets[0]
    groups = [("all", df)] + [(f"{tgt}={k}", g) for k, g in df.groupby(tgt, dropna=False, observed=True)]
else:
    groups = [("all", df)]

rows = []
for gname, sub in groups:
    for col in feature_cols:
        s = sub[col]
        n = s.notna().sum()
        n_missing = s.isna().sum()
        if col in numeric_cols:
            row = dict(group=gname, variable=col, type="numeric", n=n, n_missing=n_missing,
                       mean=s.mean(), sd=s.std(), min=s.min(), median=s.median(), max=s.max(),
                       n_levels=np.nan, top_share=np.nan)
        else:
            vc = s.value_counts(dropna=True, normalize=True)
            top_share = vc.iloc[0] if len(vc) else np.nan
            row = dict(group=gname, variable=col, type="categorical", n=n, n_missing=n_missing,
                       mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                       n_levels=s.nunique(dropna=True), top_share=top_share)
        rows.append(row)

summary = pd.DataFrame(rows, columns=["group", "variable", "type", "n", "n_missing",
                                      "mean", "sd", "min", "median", "max",
                                      "n_levels", "top_share"])
if len(summary) > 50:
    summary = summary[summary["group"] == "all"].copy()
    if len(summary) > 50:
        summary = summary.head(50)

def sig_round(x):
    if pd.isna(x):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    return float(f"{x:.4g}")

for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[c] = summary[c].map(sig_round)

mlca_runtime.save_table(summary, "distributions.csv", caption="Distribution summaries by target class")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")

print(f"features {len(feature_cols)}: numeric={len(numeric_cols)} categorical={len(cat_cols)}")
print(f"summary rows {len(summary)} cols {len(summary.columns)}")
print(f"placeholders found in {placeholder_cols} text columns")
print("saved distributions.csv")

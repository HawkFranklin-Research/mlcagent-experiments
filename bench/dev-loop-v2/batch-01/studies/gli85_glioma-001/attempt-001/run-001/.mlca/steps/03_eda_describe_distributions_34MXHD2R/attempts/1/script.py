import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["gli85_glioma_openml"])

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
_ = protocol.get("task", {})

def is_keylike(col):
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return False
    n = s.notna().sum()
    return n > 0 and s.nunique(dropna=True) == n

feature_cols = [
    c for c in df.columns
    if c not in targets and c not in excluded and c != group_col and not is_keylike(c)
]
numeric_cols = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])
]
categorical_cols = [
    c for c in feature_cols
    if c not in numeric_cols and not pd.api.types.is_datetime64_any_dtype(df[c])
]

text_cols = list(df.select_dtypes(include=["object", "string"]).columns)
found_tokens, found_cols = set(), set()
for c in text_cols:
    stripped = df[c].astype("string").str.strip()
    for token in ["?", "-", "NA", "N/A", ""]:
        if stripped.eq(token).any():
            found_tokens.add(token if token else "whitespace")
            found_cols.add(c)

def num_stats(frame, cols):
    if not cols:
        return dict(n_columns=0, mean=np.nan, sd=np.nan, min=np.nan,
                    median=np.nan, max=np.nan, n_missing=0,
                    n_levels=np.nan, top_level_share=np.nan)
    sub = frame[cols]
    means = sub.mean(numeric_only=True)
    sds = sub.std(numeric_only=True, ddof=1)
    mins = sub.min(numeric_only=True)
    meds = sub.median(numeric_only=True)
    maxs = sub.max(numeric_only=True)
    return dict(
        n_columns=len(cols),
        mean=float(np.nanmean(means)),
        sd=float(np.nanmean(sds)),
        min=float(np.nanmin(mins)),
        median=float(np.nanmedian(meds)),
        max=float(np.nanmax(maxs)),
        n_missing=int(sub.isna().sum().sum()),
        n_levels=np.nan,
        top_level_share=np.nan,
    )

def cat_stats(frame, cols):
    if not cols:
        return dict(n_columns=0, mean=np.nan, sd=np.nan, min=np.nan,
                    median=np.nan, max=np.nan, n_missing=0,
                    n_levels=np.nan, top_level_share=np.nan)
    levels, shares, missing = [], [], 0
    for c in cols:
        s = frame[c]
        vc = s.value_counts(dropna=True)
        levels.append(float(s.nunique(dropna=True)))
        shares.append(float(vc.iloc[0] / vc.sum()) if len(vc) else np.nan)
        missing += int(s.isna().sum())
    return dict(
        n_columns=len(cols),
        mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
        n_missing=missing,
        n_levels=float(np.nanmean(levels)) if levels else np.nan,
        top_level_share=float(np.nanmean(shares)) if shares else np.nan,
    )

rows = []
rows.append({"group": "overall", "variable_type": "numeric", **num_stats(df, numeric_cols)})
rows.append({"group": "overall", "variable_type": "categorical", **cat_stats(df, categorical_cols)})

for t in targets:
    if t not in df.columns:
        continue
    for value in pd.Series(df[t].dropna().unique()).sort_values().tolist():
        sub = df.loc[df[t] == value]
        g = f"{t}={value}"
        rows.append({"group": g, "variable_type": "numeric", **num_stats(sub, numeric_cols)})
        if categorical_cols:
            rows.append({"group": g, "variable_type": "categorical", **cat_stats(sub, categorical_cols)})

out = pd.DataFrame(rows)
out = out[["group", "variable_type", "n_columns", "mean", "sd", "min",
           "median", "max", "n_missing", "n_levels", "top_level_share"]]

def sig(x):
    if pd.isna(x):
        return x
    if float(x) == 0:
        return 0.0
    return float(f"{float(x):.4g}")

for c in ["mean", "sd", "min", "median", "max", "n_levels", "top_level_share"]:
    out[c] = out[c].map(sig)

mlca_runtime.save_table(out, "distributions.csv", "Distribution summary by feature type and target class.")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")

print(f"rows={len(df)}, features={len(feature_cols)}, numeric={len(numeric_cols)}, categorical={len(categorical_cols)}")
print(f"Placeholders found: {sorted(found_tokens)} in {len(found_cols)} columns")
print(f"distributions.csv rows={len(out)}")

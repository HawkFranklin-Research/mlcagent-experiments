import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["gcm_tumour_types_openml"])
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys: {missing}. Available: {list(protocol.keys())}")

targets = protocol["targets"] or []
excluded = set(protocol["excluded_columns"] or [])
group_col = protocol["group_column"]
target = targets[0] if targets else None

key_like = []
for c in df.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = df[c]
    if s.dtype.kind in "fcmM":
        continue
    if s.nunique(dropna=True) == len(s) and s.nunique(dropna=True) > 2:
        key_like.append(c)

feature_cols = [
    c for c in df.columns
    if c not in targets and c not in excluded and c != group_col and c not in key_like
]

numeric_cols = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
]
datetime_cols = [c for c in feature_cols if pd.api.types.is_datetime64_any_dtype(df[c])]
categorical_cols = [
    c for c in feature_cols
    if c not in numeric_cols and c not in datetime_cols
    and (pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(df[c]))
]

placeholders = ["?", "-", "NA", "N/A"]
found_placeholder_cols = 0
for c in categorical_cols:
    s = df[c].astype("string").str.strip()
    mask = s.isin(placeholders) | (s == "")
    if mask.any():
        found_placeholder_cols += 1
        df[c] = s.mask(mask, pd.NA)

def summarize(sub, group_label):
    rows = []
    if len(numeric_cols) == 0:
        rows.append(dict(group=group_label, type="numeric", n_columns=0,
                         mean_of_means=np.nan, mean_of_sds=np.nan,
                         min_of_mins=np.nan, median_of_medians=np.nan,
                         max_of_maxes=np.nan, n_missing=0,
                         mean_n_levels=np.nan, mean_top_share=np.nan))
    else:
        desc = sub[numeric_cols].describe().T
        n_missing = sub[numeric_cols].isna().sum().sum()
        rows.append(dict(group=group_label, type="numeric", n_columns=len(numeric_cols),
                         mean_of_means=desc["mean"].mean(),
                         mean_of_sds=desc["std"].mean(),
                         min_of_mins=desc["min"].min(),
                         median_of_medians=desc["50%"].median(),
                         max_of_maxes=desc["max"].max(),
                         n_missing=n_missing,
                         mean_n_levels=np.nan, mean_top_share=np.nan))

    if len(categorical_cols) == 0:
        rows.append(dict(group=group_label, type="categorical", n_columns=0,
                         mean_of_means=np.nan, mean_of_sds=np.nan,
                         min_of_mins=np.nan, median_of_medians=np.nan,
                         max_of_maxes=np.nan, n_missing=0,
                         mean_n_levels=np.nan, mean_top_share=np.nan))
    else:
        n_levels, top_shares, n_missing = [], [], 0
        for c in categorical_cols:
            vc = sub[c].value_counts(dropna=True)
            n_levels.append(len(vc))
            if len(vc) > 0:
                top_shares.append(vc.iloc[0] / vc.sum())
            else:
                top_shares.append(np.nan)
            n_missing += sub[c].isna().sum()
        valid_shares = [x for x in top_shares if not pd.isna(x)]
        rows.append(dict(group=group_label, type="categorical", n_columns=len(categorical_cols),
                         mean_of_means=np.nan, mean_of_sds=np.nan,
                         min_of_mins=np.nan, median_of_medians=np.nan,
                         max_of_maxes=np.nan, n_missing=n_missing,
                         mean_n_levels=np.mean(n_levels) if n_levels else np.nan,
                         mean_top_share=np.mean(valid_shares) if valid_shares else np.nan))
    return rows

rows = summarize(df, "All")

if target and target in df.columns:
    classes = df[target].dropna().unique()
    if len(classes) > 24:
        classes = df[target].value_counts().head(24).index.tolist()
    for cls in classes:
        rows.extend(summarize(df[df[target] == cls], f"{target}={cls}"))

summary_df = pd.DataFrame(rows)
cols = ["group", "type", "n_columns", "mean_of_means", "mean_of_sds",
        "min_of_mins", "median_of_medians", "max_of_maxes", "n_missing",
        "mean_n_levels", "mean_top_share"]
summary_df = summary_df[cols]
float_cols = summary_df.select_dtypes(include=[np.floating]).columns
summary_df[float_cols] = summary_df[float_cols].round(4)

mlca_runtime.save_table(summary_df, "distributions.csv", caption="Column distribution summaries")

num_row = summary_df[(summary_df["group"] == "All") & (summary_df["type"] == "numeric")]
cat_row = summary_df[(summary_df["group"] == "All") & (summary_df["type"] == "categorical")]
n_numeric = int(num_row["n_columns"].iloc[0]) if len(num_row) else 0
n_categorical = int(cat_row["n_columns"].iloc[0]) if len(cat_row) else 0
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"features: numeric={n_numeric}, categorical={n_categorical}")
if found_placeholder_cols:
    print(f"placeholders found in {found_placeholder_cols} categorical column(s)")
else:
    print("placeholders: none")
print(f"distributions rows: {len(summary_df)}")

import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
df = pd.read_parquet(paths["smk187_airway_openml"])

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns") or [])
group_col = protocol.get("group_column")
target = targets[0]

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
features = [c for c in df.columns if c not in drop]

n_rows = len(df)
keep = []
for c in features:
    s = df[c]
    if (s.dtype == object or str(s.dtype) == "category"):
        u = s.nunique(dropna=True)
        if u == n_rows and u > 50：
            continue
    keep.append(c)
features = keep

num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in features if c not in num_cols]

ph_cols = 0
ph_tokens = {"?", "-", "NA", "N/A", ""}
for c in cat_cols:
    s = df[c].astype("string").str.strip()
    if s.isin(ph_tokens).sum() > 0:
        ph_cols += 1
print(f"placeholder tokens found in {ph_cols} of {len(cat_cols)} text columns")


def agg_row(scope, stat, series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "scope": scope, "statistic": stat, "n_columns": int(s.shape[0]),
        "min": s.min() if len(s) else np.nan,
        "p25": s.quantile(0.25) if len(s) else np.nan,
        "median": s.median() if len(s) else np.nan,
        "p75": s.quantile(0.75) if len(s) else np.nan,
        "max": s.max() if len(s) else np.nan,
    }


rows = []
if num_cols:
    num_df = df[num_cols]
    stats = pd.DataFrame({
        "mean": num_df.mean(), "sd": num_df.std(), "min": num_df.min(),
        "median": num_df.median(), "max": num_df.max(),
        "n_missing": num_df.isna().sum(),
    })
    for stat in ["mean", "sd", "min", "median", "max", "n_missing"]:
        rows.append(agg_row("overall", stat, stats[stat]))

    classes = list(pd.Series(df[target]).dropna().unique())
    class_means = {}
    for cls in classes:
        cm = df.loc[df[target] == cls, num_cols].mean()
        class_means[cls] = cm
        rows.append(agg_row(f"class={cls}", "column_mean", cm))
    if len(classes) == 2:
        c0, c1 = classes
        diff = (class_means[c1] - class_means[c0]).abs()
        rows.append(agg_row("class_diff", "abs_mean_diff", diff))

if cat_cols:
    lv = df[cat_cols].nunique()
    top = df[cat_cols].apply(
        lambda s: s.value_counts(normalize=True, dropna=True).iloc[0] if s.notna().any() else np.nan
    )
    rows.append(agg_row("categorical", "n_levels", lv))
    rows.append(agg_row("categorical", "top_level_share", top))

out = pd.DataFrame(rows)
if len(out) > 50:
    out = out.head(50)
num_cols_out = ["n_columns", "min", "p25", "median", "p75", "max"]
out[num_cols_out] = out[num_cols_out].round(4)

mlca_runtime.save_table(out, "distributions.csv", caption="Per-column statistic distributions and per-class column-mean summaries")
mlca_runtime.record_fact_candidate("n_numeric", int(len(num_cols)), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(len(cat_cols)), source="distributions.csv")

print(f"rows x cols: {df.shape[0]} x {df.shape[1]}")
print(f"n_numeric={len(num_cols)} n_categorical={len(cat_cols)}")
print(f"summary rows: {len(out)}")
print("class counts: " + str(pd.Series(df[target]).value_counts().to_dict()))

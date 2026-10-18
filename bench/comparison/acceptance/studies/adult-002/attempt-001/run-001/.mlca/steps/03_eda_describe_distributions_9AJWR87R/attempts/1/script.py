import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

targets = proto["targets"]
excluded = list(proto.get("excluded_columns", []))
group_col = proto.get("group_column")

df = pd.read_parquet(inputs["adult_census_uci"])
print(f"loaded {df.shape[0]} rows, {df.shape[1]} cols")

drop = set(targets) | set(excluded)
if group_col and group_col in df.columns:
    drop.add(group_col)

feat_cols = []
for c in df.columns:
    if c in drop:
        continue
    s = df[c]
    if s.nunique(dropna=False) == len(s) and len(s) > 1:
        if s.dtype == object or pd.api.types.is_integer_dtype(s.dtype):
            continue
    feat_cols.append(c)

placeholder_tokens = {"?", "-", "NA", "N/A", ""}
ph_cols = 0
for c in feat_cols:
    if df[c].dtype == object:
        vals = df[c].astype(str).str.strip()
        if vals.isin(placeholder_tokens).any():
            ph_cols += 1
print(f"placeholder columns: {ph_cols}")

groups = [("all", df.index)]
if len(targets) == 1 and df[targets[0]].nunique() <= 10:
    tc = targets[0]
    for cls in sorted(df[tc].dropna().unique()):
        groups.append((f"{tc}={cls}", df.index[df[tc] == cls]))

rows = []
for grp_name, idx in groups:
    sub = df.loc[idx]
    for c in feat_cols:
        s = sub[c]
        n_miss = int(s.isna().sum())
        is_num = pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype)
        sv = s.dropna()
        row = {"column": c, "type": "numeric" if is_num else "categorical",
               "group": grp_name, "n": len(sv), "n_missing": n_miss,
               "mean": np.nan, "sd": np.nan, "min": np.nan,
               "median": np.nan, "max": np.nan,
               "n_levels": np.nan, "top_share": np.nan}
        if is_num and len(sv) > 0:
            row["mean"] = round(float(sv.mean()), 4)
            row["sd"] = round(float(sv.std()), 4) if len(sv) > 1 else np.nan
            row["min"] = round(float(sv.min()), 4)
            row["median"] = round(float(sv.median()), 4)
            row["max"] = round(float(sv.max()), 4)
        elif not is_num and len(sv) > 0:
            row["n_levels"] = int(sv.nunique())
            row["top_share"] = round(float(sv.value_counts(normalize=True).iloc[0]), 4)
        rows.append(row)

summary = pd.DataFrame(rows)
if len(summary) > 50:
    all_rows = summary[summary["group"] == "all"]
    other = summary[summary["group"] != "all"]
    summary = pd.concat([all_rows, other.head(max(50 - len(all_rows), 0))], ignore_index=True)

mlca_runtime.save_table(summary, "distributions.csv",
                        caption="Feature distributions by column and target class")
print(f"saved {len(summary)} summary rows, {summary.shape[1]} cols")

n_num = sum(1 for c in feat_cols
            if pd.api.types.is_numeric_dtype(df[c].dtype) and not pd.api.types.is_bool_dtype(df[c].dtype))
n_cat = len(feat_cols) - n_num
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")
print(f"numeric={n_num}, categorical={n_cat}")

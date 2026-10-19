import json
import math
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    group_col = protocol["group_column"]
    task = protocol["task"]
except KeyError as e:
    raise KeyError(f"Missing key {e} in _protocol. Available keys: {list(protocol.keys())}")

df = pd.read_parquet(inputs["dry_bean_uci"])
targets = [targets] if isinstance(targets, str) else list(targets)
excluded = [] if excluded is None else list(excluded)
group_cols = [] if group_col is None else ([group_col] if isinstance(group_col, str) else list(group_col))

all_cols = list(df.columns)
drop = set(targets) | set(excluded) | set(group_cols)
features = [c for c in all_cols if c not in drop]

n_rows = len(df)
key_like = []
for c in features:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=True) == n_rows:
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=True) == n_rows:
            key_like.append(c)
features = [c for c in features if c not in key_like]

for c in features:
    if pd.api.types.is_datetime64_any_dtype(df[c]):
        df[c] = df[c].astype("int64") // 10**9

numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in features if c not in numeric_cols]

def is_num(c):
    return c in numeric_cols

def stats_row(c, group, sub):
    s = sub[c]
    row = {"variable": c, "type": "numeric" if is_num(c) else "categorical", "group": group}
    n_notna = s.notna().sum()
    row["n"] = n_notna
    row["n_missing"] = s.isna().sum()
    if is_num(c):
        row["mean"] = s.mean()
        row["sd"] = s.std()
        row["min"] = s.min()
        row["median"] = s.median()
        row["max"] = s.max()
        row["n_levels"] = np.nan
        row["top_share"] = np.nan
    else:
        row["mean"] = np.nan
        row["sd"] = np.nan
        row["min"] = np.nan
        row["median"] = np.nan
        row["max"] = np.nan
        vc = s.value_counts(dropna=True)
        row["n_levels"] = vc.size
        row["top_share"] = vc.iloc[0] / n_notna if n_notna > 0 else np.nan
    return row

rows = []
for c in features:
    rows.append(stats_row(c, "overall", df))

if len(targets) == 1:
    tgt = targets[0]
    if tgt in df.columns:
        classes = df[tgt].dropna().unique()
        n_classes = len(classes)
        if n_classes > 0:
            max_per_class = min(4, (50 - len(rows)) // n_classes)
            if max_per_class > 0:
                vars_ = {c: df[c].var() for c in numeric_cols}
                vars_ = {k: (v if not pd.isna(v) else 0) for k, v in vars_.items()}
                selected = sorted(vars_, key=vars_.get, reverse=True)[:max_per_class]
                for c in selected:
                    for cls in classes:
                        sub = df[df[tgt] == cls]
                        if len(sub) > 0:
                            rows.append(stats_row(c, str(cls), sub))

df_sum = pd.DataFrame(rows)
cols = ["variable", "type", "group", "n", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_share"]
df_sum = df_sum[cols]

def sig_round(x):
    if pd.isna(x) or x == 0:
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + 3)

for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    df_sum[c] = df_sum[c].apply(sig_round)

mlca_runtime.save_table(df_sum, "distributions.csv", caption="Distribution summary overall and by target class.")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")

placeholders = ["?", "-", "NA", "N/A"]
cnt = 0
for c in categorical_cols:
    s = df[c].astype(str)
    if s.str.strip().isin(placeholders).any() or s.str.strip().eq("").any():
        cnt += 1
if cnt > 0:
    print(f"Placeholder tokens found in {cnt} categorical columns")

print(f"Features: {len(features)} (numeric {len(numeric_cols)}, categorical {len(categorical_cols)})")
print(f"Summary rows: {len(df_sum)}")
print("Saved distributions.csv")

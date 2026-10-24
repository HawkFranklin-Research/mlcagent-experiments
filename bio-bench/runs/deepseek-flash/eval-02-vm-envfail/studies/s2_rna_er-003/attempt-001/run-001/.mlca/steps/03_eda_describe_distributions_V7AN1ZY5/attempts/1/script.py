import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())

try:
    targets = proto["targets"]
    excluded = set(proto.get("excluded_columns", []))
    group_col = proto.get("group_column")
    task = proto["task"]
except KeyError as e:
    raise KeyError(f"Missing key {e} in _protocol. Available keys: {list(proto.keys())}")

df = pd.read_parquet(paths["rna_er"])
target = targets[0]

drop = set(targets) | excluded | ({group_col} if group_col else set())
candidates = [c for c in df.columns if c not in drop]

# Feature columns: drop key-like all-distinct integer/string columns
feature_cols = []
for c in candidates:
    s = df[c]
    nun = s.nunique(dropna=True)
    n_obs = s.notna().sum()
    if pd.api.types.is_integer_dtype(s) and nun == n_obs:
        continue
    if (s.dtype == object or pd.api.types.is_string_dtype(s)) and nun == n_obs:
        continue
    feature_cols.append(c)

numeric_cols, cat_cols, date_cols = [], [], []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    else:
        cat_cols.append(c)

# Placeholder token check (RL-0023)
text_cols = [c for c in cat_cols if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
placeholder_cols = 0
for c in text_cols:
    s = df[c].dropna().astype(str).str.strip()
    if s.isin(["?", "-", "NA", "N/A", ""]).any():
        placeholder_cols += 1
print(f"Placeholder tokens in {placeholder_cols} text columns")

def make_row(frame, col, group, typ):
    s = frame[col]
    n = int(s.notna().sum())
    n_missing = int(s.isna().sum())
    if typ == "numeric":
        mean = s.mean()
        sd = s.std()
        mn = s.min()
        med = s.median()
        mx = s.max()
        n_levels = np.nan
        top_share = np.nan
    else:
        mean = sd = mn = med = mx = np.nan
        vc = s.value_counts(dropna=True)
        n_levels = int(vc.size)
        top_share = float(vc.iloc[0] / n) if n > 0 else np.nan
    return {
        "variable": col, "group": group, "type": typ,
        "n": n, "n_missing": n_missing,
        "mean": mean, "sd": sd, "min": mn, "median": med, "max": mx,
        "n_levels": n_levels, "top_share": top_share,
    }

rows = []
for c in numeric_cols:
    rows.append(make_row(df, c, "overall", "numeric"))
for c in cat_cols:
    rows.append(make_row(df, c, "overall", "categorical"))

# Per-class rows for up to 5 numeric and 5 categorical columns
per_class_vars = numeric_cols[:5] + cat_cols[:5]
for c in per_class_vars:
    typ = "numeric" if c in numeric_cols else "categorical"
    for cls in df[target].dropna().unique():
        sub = df[df[target] == cls]
        rows.append(make_row(sub, c, f"{target}={cls}", typ))

if len(rows) > 50:
    rows = rows[:50]

summary = pd.DataFrame(rows)
for c in ["mean", "sd", "min", "median", "max", "n_levels", "top_share"]:
    summary[c] = summary[c].apply(
        lambda x: float(f"{x:.4g}") if pd.notna(x) and isinstance(x, (int, float)) else x
    )

n_numeric = summary.loc[summary["type"] == "numeric", "variable"].nunique()
n_categorical = summary.loc[summary["type"] == "categorical", "variable"].nunique()

mlca_runtime.save_table(summary, "distributions.csv",
                        caption="Summary statistics per feature column and target class.")
mlca_runtime.record_fact_candidate("n_numeric", int(n_numeric), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(n_categorical), source="distributions.csv")

print(f"Saved distributions.csv with shape {summary.shape}")
print(f"n_numeric={n_numeric}, n_categorical={n_categorical}")

import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key}; available keys: {list(protocol.keys())}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")

path = inputs["discovery"]
if path.suffix == ".parquet":
    df = pd.read_parquet(path)
elif path.suffix in (".csv", ".tsv"):
    df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
else:
    raise ValueError(f"Unsupported discovery format: {path.suffix}")

drop = set(targets) | set(excluded)
if group_col in df.columns:
    drop.add(group_col)
features = [c for c in df.columns if c not in drop]

def key_like(col):
    s = df[col]
    n_ok = int(s.notna().sum())
    if n_ok <= 1:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == n_ok
    if pd.api.types.is_integer_dtype(s):
        name = str(col).lower()
        return s.nunique(dropna=True) == n_ok and ("id" in name or "code" in name)
    return False

features = [c for c in features if not key_like(c)]
numeric_features = [
    c for c in features
    if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
]
cat_features = [
    c for c in features
    if c not in numeric_features and not pd.api.types.is_datetime64_any_dtype(df[c])
]

tokens = {"?", "-", "NA", "N/A", ""}
ph_cols = 0
for c in df.columns:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.astype("string").str.strip().isin(tokens).any():
            ph_cols += 1
print(f"placeholder columns={ph_cols}")

def r4(x):
    if x is None:
        return np.nan
    try:
        x = float(x)
    except Exception:
        return np.nan
    if np.isnan(x) or np.isinf(x):
        return np.nan
    return float(f"{x:.4g}")

cols = ["variable", "type", "group", "n", "mean", "sd", "min", "median",
        "max", "n_missing", "n_levels", "top_share"]
rows = []

def add_num(variable, s, group):
    s = pd.to_numeric(s, errors="coerce")
    rows.append({
        "variable": variable, "type": "numeric", "group": group,
        "n": int(s.notna().sum()), "mean": r4(s.mean()), "sd": r4(s.std()),
        "min": r4(s.min()), "median": r4(s.median()), "max": r4(s.max()),
        "n_missing": int(s.isna().sum()), "n_levels": np.nan, "top_share": np.nan,
    })

def add_cat(variable, s, group):
    s2 = s.dropna()
    vc = s2.value_counts(normalize=True)
    rows.append({
        "variable": variable, "type": "categorical", "group": group,
        "n": int(s2.shape[0]), "mean": np.nan, "sd": np.nan, "min": np.nan,
        "median": np.nan, "max": np.nan, "n_missing": int(df[variable].isna().sum()),
        "n_levels": int(s2.nunique()), "top_share": r4(vc.iloc[0]) if len(vc) else np.nan,
    })

for c in numeric_features[:15]:
    add_num(c, df[c], "all")
for c in cat_features[:15]:
    add_cat(c, df[c], "all")

targets_in_df = [c for c in targets if c in df.columns]
if targets_in_df:
    event_col = min(targets_in_df, key=lambda c: df[c].nunique(dropna=True))
    duration_col = next((c for c in targets_in_df if c != event_col), None)
    if duration_col is not None:
        add_num(duration_col, df[duration_col], "all")
    add_cat(event_col, df[event_col], "all")
    if duration_col is not None:
        for cls in list(df[event_col].dropna().unique())[:8]:
            add_num(duration_col, df.loc[df[event_col] == cls, duration_col],
                    f"{event_col}={cls}")

rows.append({"variable": "n_numeric", "type": "count", "group": "all",
             "n": len(numeric_features), "mean": np.nan, "sd": np.nan,
             "min": np.nan, "median": np.nan, "max": np.nan,
             "n_missing": np.nan, "n_levels": np.nan, "top_share": np.nan})
rows.append({"variable": "n_categorical", "type": "count", "group": "all",
             "n": len(cat_features), "mean": np.nan, "sd": np.nan,
             "min": np.nan, "median": np.nan, "max": np.nan,
             "n_missing": np.nan, "n_levels": np.nan, "top_share": np.nan})

dist = pd.DataFrame(rows, columns=cols).head(50)
mlca_runtime.save_table(dist, "distributions.csv",
                        caption="Distribution summaries for numeric and categorical columns.")
mlca_runtime.record_fact_candidate("n_numeric", int(len(numeric_features)), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(len(cat_features)), source="distributions.csv")

print(f"features={len(features)} numeric={len(numeric_features)} categorical={len(cat_features)}")
print(f"distributions rows={len(dist)}")

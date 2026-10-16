import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}. Available keys: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
task = protocol["task"]

df = pd.read_parquet(inputs["adult_census_uci"])
for t in targets:
    if t in df.columns and (pd.api.types.is_string_dtype(df[t]) or isinstance(df[t].dtype, pd.CategoricalDtype)):
        df[t] = df[t].astype(str).str.strip().str.rstrip(".")

feature_cols = []
for c in df.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = df[c]
    if pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=True) == len(s):
            continue
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=True) == len(s):
            continue
    feature_cols.append(c)

num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

rows = []
def add_rows(sub, group_label):
    for c in num_cols:
        s = sub[c]
        rows.append({
            "variable": c, "type": "numeric", "group": group_label,
            "n": int(s.notna().sum()), "n_missing": int(s.isna().sum()),
            "mean": s.mean(), "sd": s.std(), "min": s.min(), "median": s.median(), "max": s.max(),
            "n_levels": np.nan, "top_share": np.nan
        })
    for c in cat_cols:
        s = sub[c]
        vc = s.value_counts(dropna=True)
        top_share = (vc.iloc[0] / vc.sum()) if len(vc) else np.nan
        rows.append({
            "variable": c, "type": "categorical", "group": group_label,
            "n": int(s.notna().sum()), "n_missing": int(s.isna().sum()),
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
            "n_levels": int(s.nunique(dropna=True)), "top_share": top_share
        })

add_rows(df, "all")
if len(targets) == 1:
    t = targets[0]
    for cls, sub in df.groupby(t, dropna=False):
        add_rows(sub, f"{t}={cls}")

overall = [r for r in rows if r["group"] == "all"]
per_class = [r for r in rows if r["group"] != "all"]
rows = overall + per_class
if len(rows) > 50:
    rows = rows[:50]
out = pd.DataFrame(rows)
out = out.map(lambda x: float(f"{x:.4g}") if isinstance(x, (int, float, np.number)) else x)
mlca_runtime.save_table(out, "distributions.csv", caption="Distribution summaries")

n_numeric = len(num_cols)
n_categorical = len(cat_cols)
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

ph_tokens = {"?", "-", "NA", "N/A", ""}
ph_cols = []
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype):
        s = df[c].astype(str).str.strip()
        if s.isin(ph_tokens).any():
            ph_cols.append(c)
print(f"placeholder columns: {len(ph_cols)}")
print(f"features: {len(feature_cols)} numeric: {n_numeric} categorical: {n_categorical}")
print(f"rows in distributions.csv: {len(out)}")

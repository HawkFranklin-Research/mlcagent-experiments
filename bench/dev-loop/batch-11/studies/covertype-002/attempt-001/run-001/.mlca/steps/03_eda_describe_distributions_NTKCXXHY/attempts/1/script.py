import mlca_runtime
import json, re
from collections import defaultdict
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["covertype_uci"])
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
try:
    group_col = protocol["group_column"]
except KeyError as e:
    raise KeyError(f"Missing group_column. Available top-level keys: {sorted(protocol.keys())}") from e
task = protocol["task"]
primary_metric = task["primary_metric"]

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols, ph_vals = 0, set()
for c in df.columns:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        stripped = s.astype("string").str.strip()
        mask = stripped.isin(placeholders) & s.notna()
        if mask.any():
            ph_cols += 1
            ph_vals.update(stripped[mask].unique())
print(f"Placeholders in {ph_cols} text columns: {sorted(ph_vals)}" if ph_cols else "No placeholder tokens found in text columns.")

candidate_cols = [c for c in df.columns if c not in set(targets) | set(excluded) | ({group_col} if group_col is not None else set())]
feature_cols = []
for c in candidate_cols:
    s = df[c]
    if (pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)) and s.nunique(dropna=True) == len(df) and not pd.api.types.is_float_dtype(s.dtype):
        continue
    feature_cols.append(c)

numeric_cols, categorical_cols = [], []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s.dtype) or pd.api.types.is_bool_dtype(s.dtype):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

continuous_numeric, binary_numeric = [], []
for c in numeric_cols:
    s = df[c].dropna()
    if s.nunique() <= 2 and set(s.unique()).issubset({0, 1, True, False}):
        binary_numeric.append(c)
    else:
        continuous_numeric.append(c)

rows = []
def add_row(variable, kind, group="overall", n=np.nan, n_missing=np.nan, mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan, n_levels=np.nan, top_share=np.nan):
    rows.append(dict(variable=variable, kind=kind, group=group, n=n, n_missing=n_missing, mean=mean, sd=sd, min=min, median=median, max=max, n_levels=n_levels, top_share=top_share))

for c in continuous_numeric:
    s = df[c]
    add_row(c, "numeric", n=s.notna().sum(), n_missing=s.isna().sum(), mean=s.mean(), sd=s.std(), min=s.min(), median=s.median(), max=s.max())

groups = defaultdict(list)
for c in binary_numeric:
    groups[re.sub(r"\d+$", "", c).rstrip("_")].append(c)
for prefix, cols in groups.items():
    stacked = df[cols].stack(dropna=False)
    s = stacked.dropna()
    add_row(prefix + "_binary_group", "numeric_group", n=len(s), n_missing=stacked.isna().sum(), mean=s.mean(), sd=s.std(), min=s.min(), median=s.median(), max=s.max())

for c in categorical_cols:
    s = df[c]
    vc = s.value_counts(dropna=True, normalize=True)
    add_row(c, "categorical", n=s.notna().sum(), n_missing=s.isna().sum(), n_levels=s.nunique(dropna=True), top_share=vc.iloc[0] if len(vc) else np.nan)

for t in targets:
    s = df[t]
    vc = s.value_counts(dropna=True, normalize=True)
    add_row(t, "target", group="overall", n=s.notna().sum(), n_missing=s.isna().sum(), n_levels=s.nunique(dropna=True), top_share=vc.iloc[0] if len(vc) else np.nan)
    for cls, cnt in s.value_counts(dropna=True).items():
        add_row(t, "target", group=f"class={cls}", n=cnt, n_missing=0, top_share=cnt / len(df))

n_numeric = len(numeric_cols)
n_categorical = len(categorical_cols)
add_row("n_numeric", "fact", n=n_numeric)
add_row("n_categorical", "fact", n=n_categorical)

dist = pd.DataFrame(rows)
def round_sig(x, sig=4):
    if pd.isna(x) or x == 0:
        return x
    return float(f"{x:.{sig}g}")
for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    dist[col] = dist[col].map(round_sig)

if len(dist) > 50:
    facts = dist[dist["kind"] == "fact"]
    rest = dist[dist["kind"] != "fact"]
    dist = pd.concat([facts, rest.head(50 - len(facts))], ignore_index=True)

mlca_runtime.save_table(dist, "distributions.csv", caption="Summary distributions of numeric and categorical columns.")
mlca_runtime.record_fact_candidate("n_numeric", int(n_numeric), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(n_categorical), source="distributions.csv")
print(f"Features: {len(feature_cols)} (numeric={n_numeric}, categorical={n_categorical})")
print(f"distributions.csv: {dist.shape[0]} rows, {dist.shape[1]} columns")

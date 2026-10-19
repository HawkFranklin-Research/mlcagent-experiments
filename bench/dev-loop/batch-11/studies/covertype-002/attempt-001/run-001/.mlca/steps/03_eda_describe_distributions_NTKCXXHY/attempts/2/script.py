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
excluded = protocol.get("excluded_columns", []) or []
group_col = protocol.get("group_column")

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = 0
for c in df.select_dtypes(include=["object", "string", "category"]).columns:
    s = df[c].astype("string").str.strip()
    if s.isin(placeholders).any():
        ph_cols += 1
print(f"placeholder text cols: {ph_cols}")

feature_cols = [c for c in df.columns
                if c not in set(targets) | set(excluded) | ({group_col} if group_col else set())]

numeric_cols, categorical_cols = [], []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s.dtype) or pd.api.types.is_bool_dtype(s.dtype):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

continuous_numeric, binary_numeric = [], []
for c in numeric_cols:
    vals = df[c].dropna().unique()
    if len(vals) <= 2 and set(vals).issubset({0, 1, True, False}):
        binary_numeric.append(c)
    else:
        continuous_numeric.append(c)

def rs(x, sig=4):
    if not isinstance(x, (int, float, np.floating, np.integer)) or pd.isna(x) or x == 0:
        return x
    return float(f"{x:.{sig}g}")

rows = []
def add(**kw):
    base = {k: np.nan for k in ["variable", "kind", "group", "n", "n_missing", "mean",
                                "sd", "min", "median", "max", "n_levels", "top_share"]}
    base.update(kw)
    rows.append(base)

for c in continuous_numeric:
    s = df[c]
    add(variable=c, kind="numeric", group="overall",
        n=int(s.notna().sum()), n_missing=int(s.isna().sum()),
        mean=rs(s.mean()), sd=rs(s.std()), min=rs(s.min()),
        median=rs(s.median()), max=rs(s.max()))

groups = defaultdict(list)
for c in binary_numeric:
    groups[re.sub(r"\d+$", "", c).rstrip("_")].append(c)
for prefix, cols in groups.items():
    arr = df[cols].to_numpy(dtype=float).ravel()
    n_missing = int(np.isnan(arr).sum())
    arr = arr[~np.isnan(arr)]
    add(variable=prefix + "_binary_group", kind="numeric_group", group="overall",
        n=int(len(arr)), n_missing=n_missing,
        mean=rs(float(arr.mean())) if len(arr) else np.nan,
        sd=rs(float(arr.std())) if len(arr) else np.nan,
        min=rs(float(arr.min())) if len(arr) else np.nan,
        median=rs(float(np.median(arr))) if len(arr) else np.nan,
        max=rs(float(arr.max())) if len(arr) else np.nan)

for c in categorical_cols:
    s = df[c]
    vc = s.value_counts(dropna=True, normalize=True)
    add(variable=c, kind="categorical", group="overall",
        n=int(s.notna().sum()), n_missing=int(s.isna().sum()),
        n_levels=int(s.nunique(dropna=True)),
        top_share=rs(vc.iloc[0]) if len(vc) else np.nan)

for t in targets:
    s = df[t]
    vc = s.value_counts(dropna=True, normalize=True)
    add(variable=t, kind="target", group="overall",
        n=int(s.notna().sum()), n_missing=int(s.isna().sum()),
        n_levels=int(s.nunique(dropna=True)),
        top_share=rs(vc.iloc[0]) if len(vc) else np.nan)
    for cls, cnt in s.value_counts(dropna=True).items():
        add(variable=t, kind="target", group=f"class={cls}",
            n=int(cnt), n_missing=0, top_share=rs(cnt / len(df)))

n_numeric = len(numeric_cols)
n_categorical = len(categorical_cols)
add(variable="n_numeric", kind="fact", n=n_numeric)
add(variable="n_categorical", kind="fact", n=n_categorical)

dist = pd.DataFrame(rows)
if len(dist) > 50:
    facts = dist[dist["kind"] == "fact"]
    rest = dist[dist["kind"] != "fact"]
    dist = pd.concat([facts, rest.head(50 - len(facts))], ignore_index=True)

mlca_runtime.save_table(dist, "distributions.csv",
                        caption="Summary distributions of numeric and categorical columns.")
mlca_runtime.record_fact_candidate("n_numeric", int(n_numeric), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(n_categorical), source="distributions.csv")
print(f"numeric={n_numeric}, categorical={n_categorical}, summary_rows={len(dist)}")

import mlca_runtime
import json
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_datetime64_any_dtype

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

for key in ("targets", "excluded_columns"):
    if key not in proto:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {list(proto)}")

targets = list(proto["targets"])
excluded = set(proto["excluded_columns"])
group = proto.get("group_column", None)

df = pd.read_parquet(inp["energy_efficiency_uci"])
feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group]

num_cols, cat_cols, other_cols = [], [], []
for c in feature_cols:
    s = df[c]
    if is_numeric_dtype(s) and not is_bool_dtype(s):
        num_cols.append(c)
    elif is_datetime64_any_dtype(s):
        other_cols.append(c)
    else:
        cat_cols.append(c)

tokens = {"?", "-", "NA", "N/A", ""}
ph_cols = []
for c in cat_cols:
    vals = df[c].dropna().astype(str).str.strip()
    if vals.isin(tokens).any():
        ph_cols.append(c)
print(f"placeholder columns: {len(ph_cols)}" + (f" ({', '.join(ph_cols[:5])})" if ph_cols else ""))

rows = []
for c in num_cols:
    s = df[c]
    rows.append({
        "column": c, "type": "numeric", "n": int(s.notna().sum()), "n_missing": int(s.isna().sum()),
        "mean": s.mean(), "sd": s.std(ddof=1), "min": s.min(), "median": s.median(), "max": s.max(),
        "n_levels": np.nan, "top_level_share": np.nan,
    })
for c in cat_cols:
    s = df[c]
    vc = s.dropna().astype(str).value_counts()
    n = int(vc.sum())
    top = float(vc.iloc[0] / n) if n else np.nan
    rows.append({
        "column": c, "type": "categorical", "n": n, "n_missing": int(s.isna().sum()),
        "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
        "n_levels": int(vc.size), "top_level_share": top,
    })

summary = pd.DataFrame(rows, columns=[
    "column", "type", "n", "n_missing", "mean", "sd", "min", "median", "max",
    "n_levels", "top_level_share",
]).head(50)

for c in ["mean", "sd", "min", "median", "max", "top_level_share"]:
    summary[c] = summary[c].map(lambda x: float(f"{x:.4g}") if pd.notna(x) and np.isfinite(x) else np.nan)

mlca_runtime.save_table(summary, "distributions.csv", caption="Numeric and categorical feature distributions")
n_num = int((summary["type"] == "numeric").sum())
n_cat = int((summary["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")

print(f"rows={len(df)} features={len(feature_cols)} numeric={n_num} categorical={n_cat}")
print(f"saved distributions.csv rows={len(summary)} cols={summary.shape[1]}")

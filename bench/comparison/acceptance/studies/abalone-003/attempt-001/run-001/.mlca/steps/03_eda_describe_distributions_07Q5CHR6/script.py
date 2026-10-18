import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in proto:
        raise KeyError(f"Missing protocol key {k!r}; available keys: {list(proto.keys())}")

targets = list(proto["targets"])
excluded = set(proto["excluded_columns"] or [])
group_col = proto["group_column"]
task = proto["task"]

df = pd.read_parquet(inp["abalone_uci"])

def is_key_like(s):
    if s.nunique(dropna=True) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        name = str(s.name).lower()
        return any(t in name for t in ("id", "index", "row")) or s.is_monotonic_increasing
    return False

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]
feature_cols = [c for c in feature_cols if not is_key_like(df[c])]

ph = {"?", "-", "NA", "N/A", ""}
found = {}
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s):
        vals = s.dropna().astype(str).str.strip()
        hits = vals[vals.isin(ph)]
        if len(hits):
            found[c] = sorted(hits.unique().tolist())
tokens = sorted({t for v in found.values() for t in v})
print(f"placeholders: {tokens if tokens else 'none'} in {len(found)} columns")

def sig4(x):
    if pd.isna(x) or x == 0:
        return x
    return float(f"{x:.4g}")

def is_numeric(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)

n_numeric = sum(is_numeric(df[c]) for c in feature_cols)
n_categorical = len(feature_cols) - n_numeric

groups = [("overall", df)]
if str(task.get("type", "")).lower().startswith("class") and targets:
    t = targets[0]
    for cls, part in df.groupby(t, dropna=False):
        groups.append((f"{t}={cls}", part))

rows = []
for gname, part in groups:
    for c in feature_cols:
        s = part[c]
        n = int(s.notna().sum())
        n_missing = int(s.isna().sum())
        if is_numeric(s):
            rows.append({
                "group": gname, "variable": c, "type": "numeric",
                "n": n, "n_missing": n_missing,
                "mean": sig4(s.mean()), "sd": sig4(s.std(ddof=1)),
                "min": sig4(s.min()), "median": sig4(s.median()), "max": sig4(s.max()),
                "n_levels": np.nan, "top_share": np.nan,
            })
        else:
            vc = s.value_counts(dropna=True)
            top_share = float(vc.iloc[0] / vc.sum()) if len(vc) else np.nan
            rows.append({
                "group": gname, "variable": c, "type": "categorical",
                "n": n, "n_missing": n_missing,
                "mean": np.nan, "sd": np.nan, "min": np.nan,
                "median": np.nan, "max": np.nan,
                "n_levels": int(s.nunique(dropna=True)), "top_share": sig4(top_share),
            })

dist = pd.DataFrame(rows)[
    ["group", "variable", "type", "n", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_share"]
]
mlca_runtime.save_table(dist, "distributions.csv", caption="Column distribution summary")
mlca_runtime.record_fact_candidate("n_numeric", int(n_numeric), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(n_categorical), source="distributions.csv")
print(f"features={len(feature_cols)} numeric={n_numeric} categorical={n_categorical}")
print(f"distributions rows={len(dist)}")

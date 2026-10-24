import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["rna_pr"])

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing '{k}' in _protocol. Available keys: {list(d.keys())}")
    return d[k]

targets = list(need(protocol, "targets"))
excluded = need(protocol, "excluded_columns")
if isinstance(excluded, dict):
    excluded = excluded.get("columns") or excluded.get("examples") or []
excluded = set(excluded)
group_col = protocol.get("group_column")
task = need(protocol, "task")
target = targets[0]

def is_key_like(s):
    if len(s) == 0 or s.nunique(dropna=True) != len(s):
        return False
    return not pd.api.types.is_float_dtype(s)

feat_cols = [c for c in df.columns
             if c not in targets and c not in excluded and c != group_col
             and not is_key_like(df[c])]

num_cols, cat_cols = [], []
for c in feat_cols:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    elif s.dtype == object and s.nunique(dropna=True) <= 0.05 * len(s):
        cat_cols.append(c)
    elif pd.api.types.is_bool_dtype(s):
        num_cols.append(c)

def num_summary(s, var, grp):
    return {"variable": var, "group": grp, "type": "numeric",
            "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "median": s.median(), "max": s.max(),
            "n_missing": int(s.isna().sum()),
            "n_levels": np.nan, "top_share": np.nan}

def cat_summary(s, var, grp):
    s = s.dropna()
    n = len(s)
    vc = s.value_counts() if n else pd.Series(dtype=int)
    return {"variable": var, "group": grp, "type": "categorical",
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan,
            "max": np.nan, "n_missing": np.nan,
            "n_levels": int(s.nunique()) if n else 0,
            "top_share": float(vc.iloc[0] / n) if n else np.nan}

if num_cols:
    order = df[num_cols].var(numeric_only=True).sort_values(ascending=False).index.tolist()
else:
    order = []

rows = [num_summary(df[c], c, "all") for c in order[:40]]
rows += [cat_summary(df[c], c, "all") for c in cat_cols]

sub_df = df.dropna(subset=[target])
classes = sorted(sub_df[target].unique().tolist(), key=str) if target in df.columns else []
for c in order[:5]:
    if len(rows) >= 50:
        break
    for cls in classes:
        if len(rows) >= 50:
            break
        rows.append(num_summary(sub_df.loc[sub_df[target] == cls, c], c, str(cls)))

out = pd.DataFrame(rows[:50])
out = out[["variable", "group", "type", "mean", "sd", "min", "median",
           "max", "n_missing", "n_levels", "top_share"]]
for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[col] = out[col].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)

mlca_runtime.save_table(out, "distributions.csv",
    caption="Feature distributions overall and per target class")

mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")

print(f"features: {len(feat_cols)} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")
print(f"distributions rows saved: {len(out)}")

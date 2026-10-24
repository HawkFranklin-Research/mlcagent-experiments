import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"_protocol missing {k}; available keys: {list(d.keys())}")
    return d[k]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = need(protocol, "group_column")
_ = need(protocol, "task")

df = pd.read_parquet(inp["proteome_pr"])
excluded_set = set(excluded) if excluded else set()

def is_key_like(s):
    if pd.api.types.is_bool_dtype(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s):
        n = int(s.notna().sum())
        return n == len(s) and n > 0 and int(s.nunique(dropna=True)) == n
    return False

feature_cols = [
    c for c in df.columns
    if c not in set(targets)
    and c not in excluded_set
    and (group is None or c != group)
    and not is_key_like(df[c])
]

numeric_cols = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
]
num_set = set(numeric_cols)
categorical_cols = [c for c in feature_cols if c not in num_set]

def num_score(c):
    s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(s.var(skipna=True)) if s.notna().sum() > 1 else 0.0

numeric_sorted = sorted(numeric_cols, key=num_score, reverse=True)
categorical_sorted = sorted(categorical_cols, key=lambda c: df[c].nunique(dropna=True), reverse=True)

target_col = targets[0] if targets else None
if target_col is not None and target_col not in df.columns:
    raise KeyError(f"target {target_col!r} not found in table columns")
classes = list(pd.unique(df[target_col].dropna())) if target_col else []

per_var = 1 + max(0, len(classes))
max_vars = max(1, 50 // per_var)
chosen = []
i = j = 0
while len(chosen) < max_vars and (i < len(numeric_sorted) or j < len(categorical_sorted)):
    if i < len(numeric_sorted):
        chosen.append(numeric_sorted[i])
        i += 1
    if len(chosen) >= max_vars:
        break
    if j < len(categorical_sorted):
        chosen.append(categorical_sorted[j])
        j += 1

cols = ["variable", "kind", "target_class", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_level", "top_share"]
rows = []

def summarize(s, var, kind, tclass):
    if kind == "numeric":
        sn = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return {
            "variable": var, "kind": kind, "target_class": tclass,
            "n_missing": int(s.isna().sum()),
            "mean": sn.mean(), "sd": sn.std(), "min": sn.min(),
            "median": sn.median(), "max": sn.max(),
            "n_levels": np.nan, "top_level": np.nan, "top_share": np.nan,
        }
    vc = s.value_counts(dropna=True)
    top = vc.index[0] if len(vc) else np.nan
    share = float(vc.iloc[0] / vc.sum()) if len(vc) else np.nan
    return {
        "variable": var, "kind": kind, "target_class": tclass,
        "n_missing": int(s.isna().sum()),
        "mean": np.nan, "sd": np.nan, "min": np.nan,
        "median": np.nan, "max": np.nan,
        "n_levels": int(s.nunique(dropna=True)), "top_level": top, "top_share": share,
    }

for var in chosen:
    kind = "numeric" if var in num_set else "categorical"
    s = df[var]
    rows.append(summarize(s, var, kind, "all"))
    for cls in classes:
        rows.append(summarize(s[df[target_col] == cls], var, kind, str(cls)))

out = pd.DataFrame(rows, columns=cols)

def sig4(x):
    if pd.isna(x):
        return x
    if x == 0:
        return 0.0
    return float(f"{x:.4g}")

for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[c] = out[c].map(sig4)
for c in ["n_missing", "n_levels"]:
    out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

mlca_runtime.save_table(out, "distributions.csv", caption="Feature distribution summaries")
mlca_runtime.record_fact_candidate("n_numeric", int(len(numeric_cols)), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(len(categorical_cols)), source="distributions.csv")

print(f"rows={len(df)} features={len(feature_cols)}")
print(f"numeric={len(numeric_cols)} categorical={len(categorical_cols)}")
print(f"summary_rows={len(out)}")

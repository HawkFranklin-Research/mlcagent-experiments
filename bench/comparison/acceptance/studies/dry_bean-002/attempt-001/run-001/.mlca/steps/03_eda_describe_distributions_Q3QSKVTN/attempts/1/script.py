import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

with open(inputs["_protocol"]) as f:
    protocol = json.load(f)

def need(d, key):
    if key not in d:
        raise KeyError(f"missing protocol key {key!r}; available top-level keys: {sorted(d.keys())}")
    return d[key]

def as_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)

targets = as_list(need(protocol, "targets"))
excluded = as_list(need(protocol, "excluded_columns"))
group = protocol.get("group_column", None)

df = pd.read_parquet(inputs["dry_bean_uci"])

drop = set(targets) | set(excluded)
if group is not None and group != "":
    if isinstance(group, (list, tuple, set)):
        drop.update(group)
    else:
        drop.add(group)

def key_like(s, col):
    non = s.dropna()
    if len(non) == 0 or non.nunique() != len(non):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        name = str(col).lower()
        return any(k in name for k in ("id", "code", "key"))
    return False

feature_cols = [c for c in df.columns if c not in drop and not key_like(df[c], c)]

rows = []
for col in feature_cols:
    s = df[col]
    n = len(s)
    n_missing = int(s.isna().sum())
    non = s.dropna()
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        rows.append({
            "variable": col, "type": "numeric", "level": np.nan, "n": n,
            "n_missing": n_missing,
            "mean": non.mean() if len(non) else np.nan,
            "sd": non.std(ddof=1) if len(non) > 1 else np.nan,
            "min": non.min() if len(non) else np.nan,
            "median": non.median() if len(non) else np.nan,
            "max": non.max() if len(non) else np.nan,
            "n_levels": np.nan,
            "top_share": np.nan,
        })
    else:
        vc = non.value_counts(dropna=True)
        rows.append({
            "variable": col, "type": "categorical", "level": np.nan, "n": n,
            "n_missing": n_missing,
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
            "n_levels": int(non.nunique()),
            "top_share": float(vc.iloc[0] / len(non)) if len(non) else np.nan,
        })

for t in targets:
    if t not in df.columns:
        continue
    s = df[t]
    n = len(s)
    n_missing = int(s.isna().sum())
    non = s.dropna()
    vc = non.value_counts(dropna=True)
    n_levels = int(non.nunique())
    rows.append({
        "variable": t, "type": "categorical", "level": np.nan, "n": n,
        "n_missing": n_missing,
        "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
        "n_levels": n_levels,
        "top_share": float(vc.iloc[0] / len(non)) if len(non) else np.nan,
    })
    if 1 < n_levels <= 20:
        for lev, count in vc.items():
            rows.append({
                "variable": t, "type": "target_class", "level": str(lev), "n": int(count),
                "n_missing": 0,
                "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
                "n_levels": n_levels,
                "top_share": float(count / len(non)) if len(non) else np.nan,
            })

summary = pd.DataFrame(rows, columns=[
    "variable", "type", "level", "n", "n_missing", "mean", "sd", "min",
    "median", "max", "n_levels", "top_share"
])

def round_sig(x):
    if pd.isna(x):
        return np.nan
    return float(f"{float(x):.4g}")

for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[c] = summary[c].map(round_sig)

mlca_runtime.save_table(summary, "distributions.csv", caption="Column distributions and target class distribution")

feature_summary = summary[~summary["variable"].isin(targets)]
n_numeric = int((feature_summary["type"] == "numeric").sum())
n_categorical = int((feature_summary["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

placeholders = {"?", "-", "NA", "N/A"}
token_cols = []
found = set()
for c in df.columns:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        st = s.astype("string").str.strip()
        mask = (st.isin(list(placeholders)) | (st == "")).fillna(False)
        if mask.any():
            token_cols.append(c)
            found.update(st[mask].dropna().unique().tolist())

print(f"distributions.csv: {len(summary)} rows x {summary.shape[1]} cols")
print(f"features: {n_numeric} numeric, {n_categorical} categorical")
print(f"placeholder tokens {sorted(found)} in {len(token_cols)} text columns")

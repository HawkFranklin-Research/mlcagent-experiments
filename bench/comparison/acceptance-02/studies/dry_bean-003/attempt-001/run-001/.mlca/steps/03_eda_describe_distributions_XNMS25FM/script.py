import json
import numpy as np
import pandas as pd
import mlca_runtime
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_integer_dtype

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group_col = protocol.get("group_column")
df = pd.read_parquet(paths["dry_bean_uci"])
if not targets:
    raise ValueError("No targets declared in _protocol['targets']")
target = targets[0]
if target not in df.columns:
    raise KeyError(f"Target {target!r} not found; columns include {list(df.columns)[:10]}")

def is_key_like(s, name):
    lname = str(name).lower()
    if any(tok in lname for tok in ("id", "index", "row_id", "key", "code")):
        if s.nunique(dropna=True) == s.notna().sum():
            return True
    if is_integer_dtype(s):
        vals = s.dropna()
        if len(vals) and vals.is_unique and vals.is_monotonic_increasing:
            if (vals.max() - vals.min() + 1) == len(vals):
                return True
    return False

feature_cols = []
for c in df.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    if is_key_like(df[c], c):
        continue
    feature_cols.append(c)

numeric_cols, categorical_cols = [], []
for c in feature_cols:
    s = df[c]
    if is_datetime64_any_dtype(s):
        df[c] = (s - s.min()).dt.days
        numeric_cols.append(c)
    elif is_numeric_dtype(s):
        numeric_cols.append(c)
    else:
        if s.nunique(dropna=True) > 0.05 * len(df):
            continue
        categorical_cols.append(c)

def sig(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, np.integer)):
        return int(v)
    try:
        return float(f"{float(v):.4g}")
    except Exception:
        return v

rows = []
for c in numeric_cols:
    s = df[c]
    x = s.dropna()
    rows.append({
        "variable": c, "scope": "overall", "type": "numeric", "n": int(len(x)),
        "n_missing": int(s.isna().sum()), "mean": sig(x.mean()) if len(x) else np.nan,
        "sd": sig(x.std()) if len(x) > 1 else np.nan, "min": sig(x.min()) if len(x) else np.nan,
        "median": sig(x.median()) if len(x) else np.nan, "max": sig(x.max()) if len(x) else np.nan,
        "n_levels": np.nan, "top_level_share": np.nan,
    })

for c in categorical_cols:
    s = df[c]
    nn = s.dropna()
    vc = nn.value_counts(normalize=True)
    rows.append({
        "variable": c, "scope": "overall", "type": "categorical", "n": int(len(nn)),
        "n_missing": int(s.isna().sum()), "mean": np.nan, "sd": np.nan, "min": np.nan,
        "median": np.nan, "max": np.nan, "n_levels": int(nn.nunique()),
        "top_level_share": sig(vc.iloc[0]) if len(vc) else np.nan,
    })

s = df[target]
nn = s.dropna()
vc = nn.value_counts(normalize=True)
rows.append({
    "variable": target, "scope": "overall", "type": "categorical", "n": int(len(nn)),
    "n_missing": int(s.isna().sum()), "mean": np.nan, "sd": np.nan, "min": np.nan,
    "median": np.nan, "max": np.nan, "n_levels": int(nn.nunique()),
    "top_level_share": sig(vc.iloc[0]) if len(vc) else np.nan,
})
for cls, count in s.value_counts(dropna=True).items():
    rows.append({
        "variable": target, "scope": f"target={cls}", "type": "target_class", "n": int(count),
        "n_missing": np.nan, "mean": np.nan, "sd": np.nan, "min": np.nan,
        "median": np.nan, "max": np.nan, "n_levels": np.nan,
        "top_level_share": sig(count / len(df)),
    })

def sep_ratio(c):
    d = df[[target, c]].dropna()
    if len(d) < 2 or d[target].nunique() < 2:
        return -1.0
    m = d[c].mean()
    between = sum(len(g) * (g[c].mean() - m) ** 2 for _, g in d.groupby(target))
    total = ((d[c] - m) ** 2).sum()
    return float(between / total) if total > 0 else -1.0

n_classes = int(df[target].nunique(dropna=True))
budget = 50 - len(rows)
max_vars = max(0, budget // n_classes) if n_classes else 0
ranked = sorted(numeric_cols, key=sep_ratio, reverse=True)
for c in ranked[:min(3, max_vars)]:
    for cls, g in df.groupby(target, dropna=True):
        x = g[c].dropna()
        rows.append({
            "variable": c, "scope": f"target={cls}", "type": "numeric_by_class", "n": int(len(x)),
            "n_missing": int(g[c].isna().sum()), "mean": sig(x.mean()) if len(x) else np.nan,
            "sd": sig(x.std()) if len(x) > 1 else np.nan, "min": sig(x.min()) if len(x) else np.nan,
            "median": sig(x.median()) if len(x) else np.nan, "max": sig(x.max()) if len(x) else np.nan,
            "n_levels": np.nan, "top_level_share": np.nan,
        })

cols = ["variable", "scope", "type", "n", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_level_share"]
dist = pd.DataFrame(rows, columns=cols)
if len(dist) > 50:
    dist = dist.head(50)

n_numeric = len({r["variable"] for r in rows if r["type"] == "numeric"})
n_categorical = len({r["variable"] for r in rows if r["type"] == "categorical"} - {target})

mlca_runtime.save_table(dist, "distributions.csv", caption="Distribution summaries by variable and target class")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"rows={len(dist)} numeric={n_numeric} categorical={n_categorical}")
print(f"target={target} classes={n_classes}")

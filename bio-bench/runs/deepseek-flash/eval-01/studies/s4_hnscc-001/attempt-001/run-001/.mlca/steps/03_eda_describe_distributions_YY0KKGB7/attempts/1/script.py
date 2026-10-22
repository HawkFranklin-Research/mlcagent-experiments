import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["discovery"])

def required(d, k):
    if k not in d:
        raise KeyError(f"missing required key {k!r}; available top-level keys: {list(d.keys())}")
    return d[k]

targets = required(proto, "targets")
excluded = required(proto, "excluded_columns")
task = required(proto, "task")
group_col = proto.get("group_column")
event_col = task.get("event_column") or task.get("event_col") or proto.get("event_column")

base_exclude = set(targets) | set(excluded)
if group_col:
    base_exclude.add(group_col)
if event_col:
    base_exclude.add(event_col)

def is_keylike(s):
    if len(s) <= 1 or s.nunique(dropna=False) != len(s):
        return False
    return pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)

features = []
for c in df.columns:
    if c in base_exclude:
        continue
    if is_keylike(df[c]):
        continue
    features.append(c)

placeholders = {"?", "-", "NA", "N/A"}
ph_cols = 0
for c in features:
    s = df[c]
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        vals = s.dropna().astype("string").str.strip()
        if vals.isin(placeholders).any() or (vals.str.len() == 0).any():
            ph_cols += 1
print(f"placeholders {sorted(placeholders)} in {ph_cols} columns")

num_feats = []
cat_feats = []
for c in features:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s.dtype) or pd.api.types.is_numeric_dtype(s.dtype):
        num_feats.append(c)
    else:
        cat_feats.append(c)

n_numeric = len(num_feats)
n_categorical = len(cat_feats)

group_for_summary = None
if event_col and event_col in df.columns:
    vals = df[event_col].dropna().unique()
    if 1 < len(vals) <= 5:
        group_for_summary = event_col
elif targets:
    t = targets[0]
    if t in df.columns and 1 < df[t].nunique(dropna=True) <= 5:
        group_for_summary = t

inter = []
i = j = 0
while i < len(num_feats) or j < len(cat_feats):
    if i < len(num_feats):
        inter.append((num_feats[i], "numeric"))
        i += 1
    if j < len(cat_feats):
        inter.append((cat_feats[j], "categorical"))
        j += 1

if group_for_summary:
    groups = sorted(df[group_for_summary].dropna().unique(), key=str)
    max_feats = max(1, 50 // (1 + len(groups)))
else:
    groups = [None]
    max_feats = 50
selected = inter[:max_feats]

def clean_cat(s):
    s = s.astype("string").str.strip()
    return s.mask(s.isin(list(placeholders)) | (s.str.len() == 0), pd.NA)

def num_stats(s):
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        mask = s.notna()
        s = s.astype("int64").astype(float) / 1e9 / 86400.0
        s[~mask] = np.nan
    s = pd.to_numeric(s, errors="coerce")
    n = int(s.notna().sum())
    nm = int(s.isna().sum())
    if n == 0:
        return {"n": n, "n_missing": nm, "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan}
    return {
        "n": n,
        "n_missing": nm,
        "mean": s.mean(),
        "sd": s.std(ddof=1) if n > 1 else np.nan,
        "min": s.min(),
        "median": s.median(),
        "max": s.max(),
    }

def cat_stats(s):
    s = clean_cat(s)
    n = int(s.notna().sum())
    nm = int(s.isna().sum())
    vc = s.value_counts(dropna=True)
    nlev = int(vc.size)
    share = float(vc.iloc[0] / n) if n > 0 and nlev > 0 else np.nan
    return {"n": n, "n_missing": nm, "n_levels": nlev, "top_level_share": share}

rows = []
for c, typ in selected:
    subsets = [("overall", df[c])]
    if group_for_summary:
        for g in groups:
            subsets.append((f"{group_for_summary}={g}", df.loc[df[group_for_summary] == g, c]))
    for gname, s in subsets:
        row = {"variable": c, "type": typ, "group": gname}
        row.update(num_stats(s) if typ == "numeric" else cat_stats(s))
        rows.append(row)

cols = ["variable", "type", "group", "n", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_level_share"]
out = pd.DataFrame(rows)
for c in cols:
    if c not in out.columns:
        out[c] = np.nan
out = out[cols]
out["n"] = out["n"].astype(int)
out["n_missing"] = out["n_missing"].astype(int)
out["n_levels"] = out["n_levels"].astype(float)
for c in ["mean", "sd", "min", "median", "max", "top_level_share"]:
    out[c] = out[c].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)

mlca_runtime.save_table(out, "distributions.csv", caption="Distribution summaries")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"rows: {len(out)}, numeric: {n_numeric}, categorical: {n_categorical}")

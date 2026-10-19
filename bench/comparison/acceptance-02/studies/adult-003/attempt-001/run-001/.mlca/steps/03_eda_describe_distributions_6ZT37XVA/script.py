import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["adult_census_uci"])
protocol = json.loads(paths["_protocol"].read_text())

def req(d, key):
    if key not in d:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {sorted(d)}")
    return d[key]

targets = req(protocol, "targets")
if isinstance(targets, str):
    targets = [targets]
excluded = req(protocol, "excluded_columns")
if isinstance(excluded, str):
    excluded = [excluded]
group_col = req(protocol, "group_column")
task = req(protocol, "task")
for t in targets:
    if t not in df.columns:
        raise ValueError(f"Target {t!r} not in data columns")

def key_like(s, name):
    n = int(s.notna().sum())
    if n == 0 or s.nunique(dropna=True) != n:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s):
        v = s.dropna().sort_values()
        if len(v) > 1 and v.diff().dropna().nunique() <= 1:
            return True
        if any(k in name.lower() for k in ("id", "index", "key")):
            return True
    return False

exclude_set = set(targets) | set(excluded)
if group_col is not None:
    exclude_set.add(group_col)
features = [c for c in df.columns if c not in exclude_set and not key_like(df[c], c)]

placeholders = {"?", "-", "NA", "N/A", ""}
found = set()
ph_cols = 0
df_clean = df.copy()
for c in features:
    s = df_clean[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        ss = s.astype("string").str.strip()
        mask = ss.isin(placeholders) | ss.eq("")
        if mask.any():
            ph_cols += 1
            found.update([x for x in ss[mask].dropna().unique() if x != ""])
        df_clean[c] = ss.mask(ss.isin(placeholders) | ss.eq(""))
print(f"placeholders {sorted(found)} in {ph_cols} columns")

group_specs = [("overall", None, None)]
for t in targets:
    vals = df[t].dropna().unique()
    if 1 < len(vals) <= 10:
        for cls in sorted(vals, key=lambda x: str(x)):
            group_specs.append((f"{t}={cls}", t, cls))

def is_num(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)

def round_sig(x):
    if pd.isna(x) or x == 0:
        return x
    return float(f"{x:.4g}")

rows = []
for feat in features:
    orig = df[feat]
    typ = "numeric" if is_num(orig) else "categorical"
    for gname, tcol, tval in group_specs:
        idx = df.index if tcol is None else df.index[df[tcol] == tval]
        ss = df_clean.loc[idx, feat]
        n_missing = int(ss.isna().sum())
        n = int(ss.notna().sum())
        row = {
            "feature": feat, "type": typ, "group": gname,
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_missing": n_missing, "n_levels": np.nan,
            "top_level_share": np.nan, "n": n,
        }
        if typ == "numeric":
            row.update(
                mean=ss.mean(), sd=ss.std(), min=ss.min(),
                median=ss.median(), max=ss.max(),
            )
        else:
            vc = ss.dropna().value_counts(normalize=True)
            row["n_levels"] = int(ss.nunique(dropna=True))
            row["top_level_share"] = float(vc.iloc[0]) if len(vc) else np.nan
        rows.append(row)

summary = pd.DataFrame(rows)
if len(summary) > 50:
    overall = summary[summary["group"] == "overall"]
    rest = summary[summary["group"] != "overall"]
    summary = pd.concat([overall, rest]).head(50)
for c in ["mean", "sd", "min", "median", "max", "top_level_share"]:
    summary[c] = summary[c].map(lambda x: round_sig(x) if pd.notna(x) else np.nan)
summary = summary[[
    "feature", "type", "group", "mean", "sd", "min", "median",
    "max", "n_missing", "n_levels", "top_level_share", "n",
]]
mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distributions by target class")
n_num = int(summary.loc[summary["type"] == "numeric", "feature"].nunique())
n_cat = int(summary.loc[summary["type"] == "categorical", "feature"].nunique())
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")
print(f"distributions rows={summary.shape[0]} cols={summary.shape[1]}")
print(f"n_numeric={n_num} n_categorical={n_cat}")

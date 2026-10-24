import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
ins = mlca_runtime.inputs()
protocol = json.loads(ins["_protocol"].read_text())

need = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in need if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = protocol["excluded_columns"]
if isinstance(excluded, dict):
    raise ValueError(f"excluded_columns must be a list; got dict keys {list(excluded)}")
group_col = protocol["group_column"]

df = pd.read_parquet(ins["rna_her2"])
if not targets:
    raise ValueError("No targets in protocol")
target = targets[0]
if target not in df.columns:
    raise KeyError(f"Target {target!r} not in data columns")

found = set()
ph_cols = []
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
        s = df[c].astype("string")
        stripped = s.str.strip()
        tok_mask = stripped.isin(["?", "-", "NA", "N/A", ""])
        for tok in ["?", "-", "NA", "N/A"]:
            if (stripped == tok).any():
                found.add(tok)
        if stripped.eq("").any():
            found.add("<blank>")
        if tok_mask.any():
            ph_cols.append(c)
            df[c] = stripped.mask(tok_mask, pd.NA)
print(f"placeholders found={sorted(found)}, columns={len(ph_cols)}")

excl = set(excluded)
group_cols = {group_col} if group_col is not None else set()
feature_cols = []
for c in df.columns:
    if c in set(targets) or c in excl or c in group_cols:
        continue
    s = df[c]
    non = s.dropna()
    if len(non) == 0:
        continue
    keylike = False
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and non.nunique() == len(non):
        keylike = True
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s) and non.nunique() == len(non):
        keylike = True
    if not keylike:
        feature_cols.append(c)

numeric_cols = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(df[c])
    and not pd.api.types.is_bool_dtype(df[c])
    and not pd.api.types.is_datetime64_any_dtype(df[c])
]
max_cat = max(5, int(0.05 * len(df)))
cat_cols = [
    c for c in feature_cols
    if (
        pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_string_dtype(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(df[c])
    )
    and df[c].nunique(dropna=True) <= max_cat
]

def vk(c):
    v = df[c].var(skipna=True)
    return -np.inf if pd.isna(v) else v

def num_stats(s, feature, group):
    return dict(feature=feature, kind="numeric", group=group, n_missing=int(s.isna().sum()),
                mean=s.mean(), sd=s.std(), min=s.min(), median=s.median(), max=s.max(),
                n_levels=np.nan, top_level=np.nan, top_share=np.nan)

def cat_stats(s, feature, group):
    vc = s.value_counts(dropna=True)
    if len(vc):
        top = vc.index[0]
        share = vc.iloc[0] / vc.sum()
        nlev = int(vc.size)
    else:
        top = np.nan
        share = np.nan
        nlev = 0
    return dict(feature=feature, kind="categorical", group=group, n_missing=int(s.isna().sum()),
                mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                n_levels=nlev, top_level=top, top_share=share)

overall = [num_stats(df[c], c, "all") for c in numeric_cols]
overall += [cat_stats(df[c], c, "all") for c in cat_cols]

def trim(rows, max_n):
    nums = [r for r in rows if r["kind"] == "numeric"]
    cats = [r for r in rows if r["kind"] == "categorical"]
    nums.sort(key=lambda r: vk(r["feature"]), reverse=True)
    cats.sort(key=lambda r: (r["n_levels"] if pd.notna(r["n_levels"]) else -1), reverse=True)
    n_num = min(len(nums), max(1, int(max_n * 0.7)))
    n_cat = min(len(cats), max_n - n_num)
    n_num = min(len(nums), max_n - n_cat)
    keep = nums[:n_num] + cats[:n_cat]
    rest = nums[n_num:] + cats[n_cat:]
    keep.extend(rest[:max_n - len(keep)])
    return keep[:max_n]

if len(overall) > 40:
    overall = trim(overall, 40)

classes = sorted(df[target].dropna().unique(), key=lambda x: str(x))
per_class = []
if numeric_cols:
    top_num = sorted(numeric_cols, key=vk, reverse=True)[:5]
    for c in top_num:
        for cls in classes:
            per_class.append(num_stats(df.loc[df[target] == cls, c], c, f"{target}={cls}"))
if len(overall) + len(per_class) > 50:
    per_class = per_class[:max(0, 50 - len(overall))]

cols = ["feature", "kind", "group", "n_missing", "mean", "sd", "min", "median", "max",
        "n_levels", "top_level", "top_share"]
dist = pd.DataFrame(overall + per_class, columns=cols)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    dist[c] = dist[c].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) and isinstance(x, (int, float, np.number)) else x)

mlca_runtime.save_table(dist, "distributions.csv",
                        caption="Numeric and categorical feature distributions; group=all or target class.")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")
print(f"rows={len(df)}, features={len(feature_cols)}, numeric={len(numeric_cols)}, categorical={len(cat_cols)}")
print(f"distributions rows={len(dist)}")

import json
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_object_dtype, is_string_dtype, is_integer_dtype
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["anemia"])
proto = json.loads(inputs["_protocol"].read_text())

def need(obj, key):
    if key not in obj:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {sorted(obj.keys())}")
    return obj[key]

targets = need(proto, "targets")
targets = [targets] if isinstance(targets, str) else list(targets)
excluded = need(proto, "excluded_columns")
excluded = [excluded] if isinstance(excluded, str) else list(excluded)
group_col = proto.get("group_column")
target = targets[0] if targets else None
if target is None or target not in df.columns:
    raise KeyError(f"Target column {target!r} not found in data columns")

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

def is_key_like(s, name):
    lname = str(name).lower()
    if any(t in lname for t in ("id", "code", "key", "uuid")):
        return True
    nn = int(s.notna().sum())
    if nn > 5 and (is_object_dtype(s) or is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)):
        return int(s.dropna().nunique()) == nn
    return False

features = [c for c in df.columns if c not in drop and not is_key_like(df[c], c)]
TOKENS = {"?", "-", "NA", "N/A", ""}

def clean_cat(s):
    t = s.astype("string").str.strip()
    return t.mask(t.isin(TOKENS) | t.eq(""), pd.NA)

ph_cols = []
for c in features:
    s = df[c]
    if is_object_dtype(s) or is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        t = s.astype("string").str.strip()
        if int((t.isin(TOKENS) | t.eq("")).sum()) > 0:
            ph_cols.append(c)

kind = {}
for c in features:
    s = df[c]
    kind[c] = "numeric" if is_numeric_dtype(s) and not is_datetime64_any_dtype(s) else "categorical"

def num_row(s, var, group):
    x = pd.to_numeric(s, errors="coerce")
    v = x.dropna()
    return {
        "variable": var, "type": "numeric", "group": group, "n": int(v.size),
        "mean": v.mean() if v.size else np.nan, "sd": v.std() if v.size > 1 else np.nan,
        "min": v.min() if v.size else np.nan, "median": v.median() if v.size else np.nan,
        "max": v.max() if v.size else np.nan, "n_missing": int(x.isna().sum()),
        "n_levels": np.nan, "top_share": np.nan,
    }

def cat_row(s, var, group):
    t = clean_cat(s)
    vc = t.dropna().value_counts()
    return {
        "variable": var, "type": "categorical", "group": group, "n": int(t.notna().sum()),
        "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
        "n_missing": int(t.isna().sum()), "n_levels": int(vc.size),
        "top_share": float(vc.iloc[0] / vc.sum()) if vc.size else np.nan,
    }

def row_for(s, var, group):
    return num_row(s, var, group) if kind[var] == "numeric" else cat_row(s, var, group)

recs = []
for c in features:
    if len(recs) >= 50:
        break
    recs.append(row_for(df[c], c, "all"))

classes = [x for x in pd.Series(df[target]).dropna().unique()]
classes = sorted(classes, key=lambda x: str(x))
for cls in classes:
    sub = df[df[target] == cls]
    for c in features:
        if len(recs) >= 50:
            break
        recs.append(row_for(sub[c], c, str(cls)))
    if len(recs) >= 50:
        break

out = pd.DataFrame(recs, columns=["variable", "type", "group", "n", "mean", "sd", "min",
                                  "median", "max", "n_missing", "n_levels", "top_share"])
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[c] = out[c].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)

mlca_runtime.save_table(out, "distributions.csv", caption="Column distributions overall and by target class")
types = out.drop_duplicates("variable")["type"]
n_numeric = int((types == "numeric").sum())
n_categorical = int((types == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"placeholders={sorted(TOKENS)} columns={len(ph_cols)}")
print(f"distributions_rows={len(out)}")
print(f"n_numeric={n_numeric} n_categorical={n_categorical}")

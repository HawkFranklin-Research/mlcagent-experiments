import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["proteome_her2"])
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available keys: {list(d.keys())}")
    return d[k]

targets = need(protocol, "targets")
excluded = set(need(protocol, "excluded_columns"))
group_col = protocol.get("group_column")

feature_cols = [c for c in df.columns if c not in set(targets) and c not in excluded and c != group_col]

def is_key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        nn = s.notna().sum()
        return nn > 1 and s.nunique(dropna=True) == nn
    return False

feature_cols = [c for c in feature_cols if not is_key_like(df[c])]

placeholders = ["?", "-", "NA", "N/A"]
found = set()
n_ph_cols = 0
text_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
for c in text_cols:
    s = df[c].astype("string")
    stripped = s.str.strip()
    mask = stripped.isin(placeholders) | stripped.eq("")
    if mask.any():
        n_ph_cols += 1
        found.update(stripped[mask].dropna().unique().tolist())
    df[c] = stripped.mask(mask)

numeric_vars = []
cat_vars = []
for c in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(df[c]):
        continue
    if pd.api.types.is_numeric_dtype(df[c]):
        numeric_vars.append(c)
    else:
        cat_vars.append(c)

def summarize(s, level, typ, name):
    n_missing = int(s.isna().sum())
    n = int(s.notna().sum())
    if typ == "numeric":
        x = pd.to_numeric(s, errors="coerce")
        return {"variable": name, "level": level, "type": typ, "n": n,
                "mean": x.mean(), "sd": x.std(), "min": x.min(), "median": x.median(),
                "max": x.max(), "n_missing": n_missing, "n_levels": np.nan, "top_share": np.nan}
    vc = s.value_counts(dropna=True)
    top_share = float(vc.iloc[0] / n) if n and len(vc) else np.nan
    return {"variable": name, "level": level, "type": typ, "n": n,
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan,
            "max": np.nan, "n_missing": n_missing, "n_levels": int(len(vc)), "top_share": top_share}

overall = [summarize(df[c], "overall", "numeric", c) for c in numeric_vars]
overall += [summarize(df[c], "overall", "categorical", c) for c in cat_vars]

class_rows = []
if targets and targets[0] in df:
    t = targets[0]
    vals = df[t].dropna().unique().tolist()
    if 2 <= len(vals) <= 10:
        for c in (numeric_vars[:5] + cat_vars[:5]):
            typ = "numeric" if c in numeric_vars else "categorical"
            for v in sorted(vals, key=str):
                class_rows.append(summarize(df.loc[df[t] == v, c], f"{t}={v}", typ, c))
class_rows = class_rows[:20]

limit = max(0, 50 - len(class_rows))
rows = (overall[:limit] + class_rows)[:50]
cols = ["variable", "level", "type", "n", "mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_share"]
summary = pd.DataFrame(rows, columns=cols)
for col in ["n", "mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_share"]:
    summary[col] = pd.to_numeric(summary[col], errors="coerce").round(4)

mlca_runtime.save_table(summary, "distributions.csv", caption="Distribution summaries overall and by target class")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_vars), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_vars), source="distributions.csv")

ph_display = sorted("<blank>" if x == "" else str(x) for x in found)
print(f"rows={len(df)} features={len(feature_cols)} numeric={len(numeric_vars)} categorical={len(cat_vars)}")
print(f"placeholders={ph_display if ph_display else 'none'} in {n_ph_cols} text columns")
print(f"summary_rows={len(summary)}")

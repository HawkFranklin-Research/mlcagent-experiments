import mlca_runtime
import json
import math
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["breast_cancer_diagnostic_uci"])

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column", None)

def is_texty(s):
    return (not pd.api.types.is_numeric_dtype(s)) and (not pd.api.types.is_datetime64_any_dtype(s))

def is_key_like(s):
    if is_texty(s):
        try:
            return bool(len(s) > 0 and s.nunique() == len(s))
        except Exception:
            return False
    if s.dtype.kind in "iu":
        return bool(len(s) > 0 and s.is_unique)
    return False

exclude = set(targets) | set(excluded)
if group_col:
    exclude.add(group_col)
features = [c for c in df.columns if c not in exclude and not is_key_like(df[c])]

ph_tokens = {"?", "-", "NA", "N/A"}
text_cols = [c for c in df.columns if is_texty(df[c])]
ph_cols = []
for c in text_cols:
    s = df[c].dropna().astype(str)
    stripped = s.str.strip()
    hit = stripped.isin(ph_tokens) | (stripped == "") | (s != stripped)
    n_hit = int(hit.sum())
    if n_hit > 0:
        ph_cols.append((c, n_hit))
if ph_cols:
    print("placeholder/whitespace columns: " + ", ".join(f"{c}:{n}" for c, n in ph_cols))
else:
    print("no placeholder tokens in text columns")

num_feats = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
cat_feats = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]

def sf4(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    if math.isnan(v) or math.isinf(v) or v == 0.0:
        return v
    d = 3 - int(math.floor(math.log10(abs(v))))
    return round(v, d)

def num_row(name, s):
    v = pd.to_numeric(s, errors="coerce")
    ok = bool(v.notna().any())
    return {
        "variable": name, "kind": "numeric",
        "n": int(v.notna().sum()), "n_missing": int(v.isna().sum()),
        "mean": sf4(v.mean()) if ok else np.nan,
        "sd": sf4(v.std()) if ok else np.nan,
        "min": sf4(v.min()) if ok else np.nan,
        "median": sf4(v.median()) if ok else np.nan,
        "max": sf4(v.max()) if ok else np.nan,
        "n_levels": np.nan, "top_level_share": np.nan,
    }

def cat_row(name, s, kind="categorical"):
    vc = s.value_counts(dropna=True)
    n_levels = int(vc.size)
    tot = int(vc.sum())
    top = float(vc.iloc[0] / tot) if tot > 0 else np.nan
    return {
        "variable": name, "kind": kind,
        "n": int(s.notna().sum()), "n_missing": int(s.isna().sum()),
        "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
        "n_levels": n_levels,
        "top_level_share": sf4(top) if not (isinstance(top, float) and math.isnan(top)) else np.nan,
    }

rows = [num_row(c, df[c]) for c in num_feats]
rows += [cat_row(c, df[c]) for c in cat_feats]

for t in targets:
    if t not in df.columns:
        continue
    s = df[t]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 20:
        continue
    vc = s.value_counts(dropna=True)
    rows.append(cat_row(t, s))
    if len(rows) + vc.size <= 50:
        for lvl, cnt in vc.items():
            rows.append({
                "variable": f"{t}={lvl}", "kind": "class",
                "n": int(cnt), "n_missing": 0,
                "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
                "n_levels": np.nan, "top_level_share": np.nan,
            })

summary = pd.DataFrame(rows).head(50)
mlca_runtime.save_table(summary, "distributions.csv")

n_numeric = int((summary["kind"] == "numeric").sum())
n_categorical = int((summary["kind"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"summary rows: {len(summary)}")
print(f"n_numeric: {n_numeric}, n_categorical: {n_categorical}")

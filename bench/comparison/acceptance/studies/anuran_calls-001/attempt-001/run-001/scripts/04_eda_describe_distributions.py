import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {list(d)}")
    return d[k]

targets = list(need(proto, "targets"))
excluded = list(need(proto, "excluded_columns"))
group = need(proto, "group_column")
task = need(proto, "task")
df = pd.read_parquet(inp["anuran_calls_uci"])
for t in targets:
    if t not in df.columns:
        raise KeyError(f"Target {t} not in columns {list(df.columns)}")

reserved = set(targets) | set(excluded) | ({group} if group else set())
candidates = [c for c in df.columns if c not in reserved]
keylike = []
for c in candidates:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        v = s.dropna()
        if len(v) == len(s) and v.nunique() == len(v):
            keylike.append(c)
    elif pd.api.types.is_integer_dtype(s):
        v = s.dropna()
        if len(v) == len(s) and v.nunique() == len(v):
            keylike.append(c)
features = [c for c in candidates if c not in keylike]
numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
cat_cols = [c for c in features if c not in numeric_cols]

placeholders = {"?", "-", "NA", "N/A", ""}
def clean_cat(s):
    s = s.astype("string").str.strip()
    return s.mask(s.isin(placeholders) | s.eq(""))

ph_cols, found = [], set()
for c in cat_cols:
    raw = df[c].astype("string")
    stripped = raw.str.strip()
    pl_mask = stripped.isin(placeholders) | stripped.str.fullmatch(r"\s*")
    if pl_mask.fillna(False).any():
        ph_cols.append(c)
        found.update(stripped[pl_mask.fillna(False)].dropna().unique().tolist())

rows = []
for c in numeric_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    rows.append({"column": c, "kind": "numeric", "target_class": np.nan,
                 "mean": s.mean(), "sd": s.std(), "min": s.min(), "median": s.median(),
                 "max": s.max(), "n_missing": int(s.isna().sum()), "n_levels": np.nan,
                 "top_level": np.nan, "top_share": np.nan})
for c in cat_cols:
    s = clean_cat(df[c])
    vc = s.value_counts(dropna=True)
    n = len(vc)
    top = vc.index[0] if n else np.nan
    share = vc.iloc[0] / vc.sum() if n and vc.sum() > 0 else np.nan
    rows.append({"column": c, "kind": "categorical", "target_class": np.nan,
                 "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan,
                 "max": np.nan, "n_missing": int(s.isna().sum()), "n_levels": int(n),
                 "top_level": top, "top_share": share})

if len(targets) == 1:
    t = targets[0]
    vc = df[t].value_counts(dropna=True)
    if len(rows) + len(vc) <= 50:
        total = vc.sum()
        for cls, count in vc.items():
            rows.append({"column": t, "kind": "target_class", "target_class": str(cls),
                         "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan,
                         "max": np.nan, "n_missing": np.nan, "n_levels": int(len(vc)),
                         "top_level": str(cls), "top_share": count / total if total else np.nan})

cols = ["column", "kind", "target_class", "mean", "sd", "min", "median",
        "max", "n_missing", "n_levels", "top_level", "top_share"]
out = pd.DataFrame(rows, columns=cols)
if len(out) > 50:
    out = out.iloc[:50].copy()

def rs(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(f"{float(x):.4g}")
    except Exception:
        return x
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[c] = out[c].map(rs)

mlca_runtime.save_table(out, "distributions.csv", "Distribution summaries")
mlca_runtime.record_fact_candidate("n_numeric", int((out["kind"] == "numeric").sum()), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int((out["kind"] == "categorical").sum()), source="distributions.csv")

print(f"features={len(features)} numeric={len(numeric_cols)} categorical={len(cat_cols)}")
print(f"placeholder columns: {len(ph_cols)}; tokens={sorted(found)}")
print(f"distributions rows={len(out)}")

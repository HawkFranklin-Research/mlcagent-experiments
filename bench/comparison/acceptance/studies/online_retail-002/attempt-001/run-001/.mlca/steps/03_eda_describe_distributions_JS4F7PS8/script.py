import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def req(d, key, where="_protocol"):
    if key not in d:
        raise KeyError(f"Missing {key!r} in {where}; available keys: {sorted(d.keys())}")
    return d[key]

targets = list(req(protocol, "targets"))
excluded = list(req(protocol, "excluded_columns"))
group_col = req(protocol, "group_column")
task = req(protocol, "task")
_ = task["primary_metric"]

df = pd.read_parquet(inputs["online_retail_uci"])
drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

def keylike(s):
    if s.isna().any():
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

feature_cols = [c for c in df.columns if c not in drop]
feature_cols = [c for c in feature_cols if not keylike(df[c])]

work = df.copy()
tokens = {"?", "-", "NA", "N/A"}
token_cols = {t: set() for t in tokens}
token_cols["whitespace"] = set()
for c in feature_cols:
    s = work[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype("string").str.strip()
        for t in tokens:
            mask = stripped.eq(t) & s.notna()
            if mask.any():
                token_cols[t].add(c)
        ws = stripped.eq("") & s.notna()
        if ws.any():
            token_cols["whitespace"].add(c)
        work[c] = stripped.mask(stripped.isin(tokens) | stripped.eq(""), pd.NA)

placeholder_cols = set().union(*token_cols.values()) if token_cols else set()
found = [f"{t}({len(cols)})" for t, cols in token_cols.items() if cols]
placeholder_line = "placeholders: " + (", ".join(found) if found else "none") + f"; cols={len(placeholder_cols)}"

rows = []
for c in feature_cols:
    s = work[c]
    n_missing = int(s.isna().sum())
    if pd.api.types.is_bool_dtype(s):
        kind = "categorical"
    elif pd.api.types.is_numeric_dtype(s):
        kind = "numeric"
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        kind = "categorical"
    else:
        continue
    if kind == "numeric":
        rows.append({
            "variable": c, "kind": kind, "n_missing": n_missing,
            "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "median": s.median(), "max": s.max(),
            "n_levels": np.nan, "top_level_share": np.nan,
        })
    else:
        vc = s.value_counts(normalize=True, dropna=True)
        rows.append({
            "variable": c, "kind": kind, "n_missing": n_missing,
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_levels": int(s.nunique(dropna=True)),
            "top_level_share": float(vc.iloc[0]) if len(vc) else np.nan,
        })

summary = pd.DataFrame(rows, columns=[
    "variable", "kind", "n_missing", "mean", "sd", "min", "median",
    "max", "n_levels", "top_level_share"
])
if len(summary) > 50:
    summary = summary.head(50)

def sig4(x):
    if pd.isna(x):
        return np.nan
    return float(f"{float(x):.4g}")

for c in ["mean", "sd", "min", "median", "max", "top_level_share"]:
    summary[c] = summary[c].map(sig4)

n_numeric = int((summary["kind"] == "numeric").sum())
n_categorical = int((summary["kind"] == "categorical").sum())

mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distribution summary")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"rows={len(df)}, features={len(feature_cols)}")
print(placeholder_line)
print(f"numeric={n_numeric}, categorical={n_categorical}")
print(f"saved distributions.csv rows={len(summary)}")

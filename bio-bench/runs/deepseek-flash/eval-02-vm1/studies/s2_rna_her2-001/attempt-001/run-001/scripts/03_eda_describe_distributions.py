import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["rna_her2"])

def required(d, key):
    if key not in d:
        raise KeyError(f"{key} missing; available top-level keys: {list(d)}")
    return d[key]

targets = required(protocol, "targets")
excluded_raw = required(protocol, "excluded_columns")
if isinstance(excluded_raw, dict):
    excluded_list = None
    for key in ("entries", "columns", "all", "values", "list"):
        if isinstance(excluded_raw.get(key), list):
            excluded_list = excluded_raw[key]
            break
    if excluded_list is None:
        excluded_list = excluded_raw.get("examples", [])
else:
    excluded_list = excluded_raw
excluded = set(excluded_list if isinstance(excluded_list, list) else [])
group_col = protocol.get("group_column")

base_cols = [c for c in df.columns if c not in set(targets) and c not in excluded and c != group_col]
feature_cols = []
for c in base_cols:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    name = c.lower()
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == len(s) and len(s) > 1:
        continue
    if any(t in name for t in ("id", "sample", "patient", "code", "barcode", "name")) and s.nunique(dropna=True) == len(s):
        continue
    feature_cols.append(c)

placeholders = {"?", "-", "NA", "N/A", ""}
placeholder_count = 0
placeholder_cols = 0
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
        continue
    s = df[c].astype("string").str.strip()
    mask = s.isin(placeholders)
    if mask.any():
        placeholder_count += int(mask.sum())
        placeholder_cols += 1
        df[c] = s.mask(mask, pd.NA)
    else:
        df[c] = s
if placeholder_count:
    print(f"placeholders={sorted(placeholders)} cols={placeholder_cols} n={placeholder_count}")

rows = []
def add_summary_row(col, typ, group):
    s = df[col]
    if typ == "numeric":
        v = pd.to_numeric(s, errors="coerce")
        rows.append({
            "variable": col, "type": "numeric", "group": group,
            "mean": v.mean(), "sd": v.std(), "min": v.min(), "median": v.median(),
            "max": v.max(), "n_missing": int(v.isna().sum()),
            "n_levels": pd.NA, "top_level": pd.NA, "top_share": pd.NA,
        })
    else:
        v = s.dropna().astype("string").str.strip()
        n = len(v)
        n_levels = v.nunique()
        if n_levels:
            counts = v.value_counts()
            top_level = counts.index[0]
            top_share = counts.iloc[0] / n
        else:
            top_level = pd.NA
            top_share = pd.NA
        rows.append({
            "variable": col, "type": "categorical", "group": group,
            "mean": pd.NA, "sd": pd.NA, "min": pd.NA, "median": pd.NA, "max": pd.NA,
            "n_missing": int(s.isna().sum()), "n_levels": int(n_levels),
            "top_level": top_level, "top_share": top_share,
        })

for c in feature_cols:
    typ = "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical"
    add_summary_row(c, typ, "all")

target = targets[0]
if target in df:
    for c in feature_cols:
        typ = "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical"
        for cls in sorted(df[target].dropna().astype(str).unique()):
            add_summary_row(c, typ, f"{target}={cls}")

summary = pd.DataFrame(rows)
if len(summary) > 50:
    summary = summary.iloc[:50].copy()
for c in ("mean", "sd", "min", "median", "max", "top_share"):
    if c in summary.columns:
        summary[c] = summary[c].map(lambda x: float(f"{x:.4g}") if pd.notna(x) and isinstance(x, (int, float)) else x)
summary = summary[["variable", "type", "group", "mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_level", "top_share"]]
mlca_runtime.save_table(summary, "distributions.csv", caption="Numeric and categorical feature summaries by target class")

n_numeric = int(summary.loc[summary["type"] == "numeric", "variable"].nunique())
n_categorical = int(summary.loc[summary["type"] == "categorical", "variable"].nunique())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"rows={len(summary)} numeric={n_numeric} categorical={n_categorical}")

import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["adult_census_uci"])

def need(key):
    if key not in protocol:
        raise ValueError(f"Missing protocol key {key!r}; available: {sorted(protocol.keys())}")
    return protocol[key]

targets = need("targets")
excluded = need("excluded_columns")
if isinstance(targets, str):
    targets = [targets]
if isinstance(excluded, str):
    excluded = [excluded]
group_col = protocol.get("group_column")

exclude_set = set(targets) | set(excluded)
if group_col:
    exclude_set.add(group_col)
base_cols = [c for c in df.columns if c not in exclude_set]

placeholder_cols = []
for c in base_cols:
    if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype):
        s = df[c].astype("string").str.strip()
        mask = s.isin(["?", "-", "NA", "N/A", ""])
        if mask.any():
            placeholder_cols.append(c)
            df[c] = s.where(~mask, pd.NA)
        else:
            df[c] = s

def is_key_like(s):
    n = len(s)
    if n == 0:
        return False
    if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
        nonmiss = s.dropna()
        return len(nonmiss) == n and nonmiss.is_unique
    if pd.api.types.is_integer_dtype(s):
        vals = s.dropna()
        if len(vals) == n and vals.is_unique and len(vals) > 1:
            sv = vals.sort_values()
            return (sv.diff().dropna() == 1).all()
    return False

features = [c for c in base_cols if not is_key_like(df[c])]
numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
categorical_cols = [c for c in features if c not in numeric_cols]

def sig4(x):
    if pd.isna(x):
        return x
    try:
        return float(f"{float(x):.4g}")
    except Exception:
        return x

def summarize(col, group_name, sub):
    s = sub[col]
    n = len(s)
    n_missing = int(s.isna().sum())
    if col in numeric_cols:
        ss = pd.to_numeric(s, errors="coerce")
        return dict(group=group_name, variable=col, type="numeric", n=n, n_missing=n_missing,
                    mean=sig4(ss.mean()), sd=sig4(ss.std()), min=sig4(ss.min()),
                    median=sig4(ss.median()), max=sig4(ss.max()), n_levels=pd.NA, top_share=pd.NA)
    ss = s.dropna().astype(str)
    n_levels = int(ss.nunique())
    top_share = sig4(ss.value_counts(normalize=True).iloc[0]) if len(ss) else pd.NA
    return dict(group=group_name, variable=col, type="categorical", n=n, n_missing=n_missing,
                mean=pd.NA, sd=pd.NA, min=pd.NA, median=pd.NA, max=pd.NA,
                n_levels=n_levels, top_share=top_share)

groups = [("All", df)]
target_col = targets[0] if targets else None
if target_col and target_col in df:
    vals = df[target_col].dropna().unique()
    try:
        vals = sorted(vals)
    except Exception:
        vals = list(vals)
    for v in vals:
        groups.append((f"{target_col}={v}", df[df[target_col] == v]))

rows = []
for gname, sub in groups:
    for c in sorted(features):
        rows.append(summarize(c, gname, sub))

if len(rows) > 50:
    all_rows = [r for r in rows if r["group"] == "All"]
    other = [r for r in rows if r["group"] != "All"]
    rows = (all_rows + other)[:50]

cols = ["group", "variable", "type", "n", "n_missing", "mean", "sd", "min",
        "median", "max", "n_levels", "top_share"]
out = pd.DataFrame(rows, columns=cols)
mlca_runtime.save_table(out, "distributions.csv", caption="Column distribution summaries by target class")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")

print(f"Loaded rows={len(df)}, columns={len(df.columns)}")
print(f"Placeholders found in {len(placeholder_cols)} text columns")
print(f"Features: numeric={len(numeric_cols)}, categorical={len(categorical_cols)}")
print(f"Saved distributions.csv rows={len(out)}")

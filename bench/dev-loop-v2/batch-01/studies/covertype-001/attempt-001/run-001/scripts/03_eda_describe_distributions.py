import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
prot = json.loads(inp["_protocol"].read_text())

for k in ["targets", "excluded_columns", "task"]:
    if k not in prot:
        raise KeyError(f"Missing protocol key {k!r}; available keys: {list(prot.keys())}")

targets = prot["targets"]
excluded = prot["excluded_columns"] or []
group_col = prot.get("group_column") if "group_column" in prot else None

df = pd.read_parquet(inp["covertype_uci"])
feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]

key_like = []
for c in feature_cols:
    s = df[c]
    if s.nunique(dropna=False) == len(df):
        name = c.lower()
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            key_like.append(c)
        elif pd.api.types.is_integer_dtype(s) and ("id" in name or "key" in name or "code" in name):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

placeholders = ["?", "-", "NA", "N/A", ""]
found = {}
for c in feature_cols:
    if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
        s = df[c].astype("string").str.strip()
        m = s.isin(["?", "-", "NA", "N/A"]) | s.eq("")
        if m.any():
            found[c] = int(m.sum())
print(f"Placeholders found: {len(found)} columns" + (f" {list(found.items())[:3]}" if found else ""))

numeric_cols, categorical_cols = [], []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

def num_stats(s):
    s = pd.to_numeric(s, errors="coerce")
    return dict(mean=s.mean(), sd=s.std(), min=s.min(), median=s.median(), max=s.max(),
                n_missing=int(s.isna().sum()))

def cat_stats(s):
    s = s.astype("string").str.strip()
    s = s.mask(s.isin(["?", "-", "NA", "N/A", ""]))
    n = int(s.notna().sum())
    top = s.value_counts(dropna=True)
    return dict(n_levels=int(s.nunique(dropna=True)),
                top_share=(top.iloc[0] / n if n and len(top) else np.nan))

def make_row(c, kind, group, n_columns, **kw):
    d = dict(column=c, kind=kind, group=str(group), n_columns=int(n_columns),
             mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
             n_missing=np.nan, n_levels=np.nan, top_share=np.nan)
    d.update(kw)
    return d

rows = []
binary = [c for c in numeric_cols if df[c].dropna().isin([0, 1]).all() and df[c].nunique(dropna=True) <= 2]
non_bin = [c for c in numeric_cols if c not in binary]

if len(binary) > 5:
    sts = [num_stats(df[c]) for c in binary]
    rows.append(make_row(f"binary_indicators (n={len(binary)})", "numeric", "overall", len(binary),
        mean=np.mean([x["mean"] for x in sts if pd.notna(x["mean"])] or [np.nan]),
        sd=np.mean([x["sd"] for x in sts if pd.notna(x["sd"])] or [np.nan]),
        min=np.min([x["min"] for x in sts if pd.notna(x["min"])] or [np.nan]),
        median=np.median([x["median"] for x in sts if pd.notna(x["median"])] or [np.nan]),
        max=np.max([x["max"] for x in sts if pd.notna(x["max"])] or [np.nan]),
        n_missing=int(sum(x["n_missing"] for x in sts))))
    for c in non_bin:
        rows.append(make_row(c, "numeric", "overall", 1, **num_stats(df[c])))
else:
    for c in numeric_cols:
        rows.append(make_row(c, "numeric", "overall", 1, **num_stats(df[c])))

if len(categorical_cols) > 20:
    sts = [cat_stats(df[c]) for c in categorical_cols]
    rows.append(make_row(f"categorical_columns (n={len(categorical_cols)})", "categorical", "overall", len(categorical_cols),
        n_levels=int(np.mean([x["n_levels"] for x in sts]) if sts else 0),
        top_share=np.mean([x["top_share"] for x in sts if pd.notna(x["top_share"])] or [np.nan])))
else:
    for c in categorical_cols:
        rows.append(make_row(c, "categorical", "overall", 1, **cat_stats(df[c])))

if len(rows) > 50:
    rows = []
    if numeric_cols:
        sts = [num_stats(df[c]) for c in numeric_cols]
        rows.append(make_row(f"numeric_columns (n={len(numeric_cols)})", "numeric", "overall", len(numeric_cols),
            mean=np.mean([x["mean"] for x in sts if pd.notna(x["mean"])] or [np.nan]),
            sd=np.mean([x["sd"] for x in sts if pd.notna(x["sd"])] or [np.nan]),
            min=np.min([x["min"] for x in sts if pd.notna(x["min"])] or [np.nan]),
            median=np.median([x["median"] for x in sts if pd.notna(x["median"])] or [np.nan]),
            max=np.max([x["max"] for x in sts if pd.notna(x["max"])] or [np.nan]),
            n_missing=int(sum(x["n_missing"] for x in sts))))
    if categorical_cols:
        sts = [cat_stats(df[c]) for c in categorical_cols]
        rows.append(make_row(f"categorical_columns (n={len(categorical_cols)})", "categorical", "overall", len(categorical_cols),
            n_levels=int(np.mean([x["n_levels"] for x in sts]) if sts else 0),
            top_share=np.mean([x["top_share"] for x in sts if pd.notna(x["top_share"])] or [np.nan])))

target_col = targets[0]
if target_col in df.columns:
    classes = sorted(df[target_col].dropna().unique())
    cont = sorted(non_bin, key=lambda c: df[c].var(skipna=True) if pd.notna(df[c].var(skipna=True)) else -1, reverse=True)
    if not cont:
        cont = numeric_cols[:3]
    for c in cont:
        for cls in classes:
            if len(rows) >= 50:
                break
            sub = df[df[target_col] == cls]
            rows.append(make_row(c, "numeric", cls, 1, **num_stats(sub[c])))
        if len(rows) >= 50:
            break

res = pd.DataFrame(rows)
for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    if col in res.columns:
        res[col] = res[col].map(lambda x: float(f"{x:.4g}") if pd.notna(x) and isinstance(x, (int, float, np.floating)) else x)
cols = ["column", "kind", "group", "n_columns", "mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_share"]
res = res[cols]

mlca_runtime.save_table(res, "distributions.csv", caption="Feature distribution summary by column type and target class where useful")

n_numeric = int(res.loc[(res.kind == "numeric") & (res.group == "overall"), "n_columns"].sum())
n_categorical = int(res.loc[(res.kind == "categorical") & (res.group == "overall"), "n_columns"].sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"rows={len(df)}, features={len(feature_cols)}, numeric={n_numeric}, categorical={n_categorical}")
print(f"distributions rows={len(res)}")
print("saved distributions.csv")

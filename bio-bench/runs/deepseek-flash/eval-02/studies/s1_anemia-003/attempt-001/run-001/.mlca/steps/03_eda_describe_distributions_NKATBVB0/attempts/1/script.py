import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
prot = json.loads(inp["_protocol"].read_text())

targets = list(prot["targets"])
excluded = list(prot.get("excluded_columns", []))
group = prot.get("group_column", None)
df = pd.read_parquet(inp["anemia"])

# feature columns (RL-0007)
feature_cols = [c for c in df.columns
                if c not in targets and c not in excluded and c != group]

# key-like columns
def is_key_like(s):
    if s.isna().all():
        return False
    if pd.api.types.is_integer_dtype(s) and s.notna().all() and s.is_unique:
        d = s.sort_index().diff().dropna()
        return d.nunique() == 1 and d.iloc[0] in (1, -1)
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == len(s)
    return False

feature_cols = [c for c in feature_cols if not is_key_like(df[c])]

# placeholder cleanup (RL-0023)
ph_cols = 0
for c in feature_cols:
    s = df[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        t = s.astype("string").str.strip()
        mask = t.isna() | t.str.fullmatch(r"\s*") | t.isin(["?", "-", "NA", "N/A", "n/a", "na"])
        if mask.sum() > s.isna().sum():
            ph_cols += 1
        df[c] = t.mask(mask)

# datetime -> days (RL-0010)
for c in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(df[c]):
        df[c] = (df[c] - pd.Timestamp("1970-01-01")).dt.days

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

def make_row(scope, col, typ, s):
    n_miss = int(s.isna().sum())
    n = int(len(s) - n_miss)
    row = dict(scope=scope, column=col, type=typ, n=n, n_missing=n_miss,
               mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
               n_levels=np.nan, top_share=np.nan)
    if typ == "numeric":
        v = pd.to_numeric(s, errors="coerce").dropna()
        if len(v):
            row["mean"] = float(v.mean())
            row["sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            row["min"] = float(v.min()); row["median"] = float(v.median()); row["max"] = float(v.max())
    else:
        v = s.dropna().astype(str)
        if len(v):
            row["n_levels"] = int(v.nunique())
            row["top_share"] = float(v.value_counts(normalize=True).iloc[0])
    return row

rows = []
for c in numeric_cols:
    rows.append(make_row("overall", c, "numeric", df[c]))
for c in categorical_cols:
    rows.append(make_row("overall", c, "categorical", df[c]))
y = df[targets[0]]
for cls in list(pd.unique(y.dropna())):
    sub = df[y == cls]
    for c in numeric_cols:
        rows.append(make_row(f"class={cls}", c, "numeric", sub[c]))
    for c in categorical_cols:
        rows.append(make_row(f"class={cls}", c, "categorical", sub[c]))

out = pd.DataFrame(rows).head(50)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[c] = out[c].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)
mlca_runtime.save_table(out, "distributions.csv",
                        caption="Numeric and categorical summaries overall and by target class")

overall = out[out["scope"] == "overall"]
n_num = int((overall["type"] == "numeric").sum())
n_cat = int((overall["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")
print(f"rows={len(df)} features={len(feature_cols)} numeric={n_num} categorical={n_cat}")
print(f"placeholder_columns={ph_cols}")
print(f"summary_rows={len(out)}")

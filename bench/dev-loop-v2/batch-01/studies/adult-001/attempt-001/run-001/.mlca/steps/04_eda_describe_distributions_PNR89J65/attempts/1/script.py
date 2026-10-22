import json
import math
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")

df = pd.read_parquet(inputs["adult_census_uci"])
target = protocol["targets"][0]
excluded = set(protocol["targets"]) | set(protocol["excluded_columns"])
group = protocol.get("group_column")
if group is not None:
    excluded.add(group)

features = [c for c in df.columns if c not in excluded]
n_rows = len(df)

def key_like(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
        return n_rows > 1 and s.notna().sum() == n_rows and s.nunique(dropna=True) == n_rows
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=True) == n_rows and n_rows > 1:
        name = str(s.name).lower()
        return "id" in name or (s.max() - s.min() + 1 == n_rows and s.is_unique)
    return False

features = [c for c in features if not key_like(df[c])]
text_cols = [c for c in features if df[c].dtype == object or pd.api.types.is_string_dtype(df[c].dtype)]
df_clean = df.copy()
placeholder_cols = 0
for c in text_cols:
    s = df_clean[c].astype("string").str.strip()
    mask = s.isin(["?", "-", "NA", "N/A"]) | (s == "")
    if mask.any():
        placeholder_cols += 1
        df_clean[c] = s.mask(mask, pd.NA)
    else:
        df_clean[c] = s

kind = {c: ("numeric" if pd.api.types.is_numeric_dtype(df_clean[c].dtype) else "categorical") for c in features}
y = df_clean[target].astype("string").str.strip().str.rstrip(".")
classes = sorted(y.dropna().unique().tolist())

def row_for(var, scope, cls, s):
    r = {
        "variable": var, "kind": kind[var], "scope": scope, "target_class": cls,
        "n_missing": int(s.isna().sum()), "mean": np.nan, "sd": np.nan,
        "min": np.nan, "median": np.nan, "max": np.nan,
        "n_levels": np.nan, "top_share": np.nan,
    }
    if kind[var] == "numeric":
        x = pd.to_numeric(s, errors="coerce")
        r.update(mean=x.mean(), sd=x.std(ddof=1), min=x.min(), median=x.median(), max=x.max())
    else:
        vc = s.dropna().value_counts(normalize=True)
        r["n_levels"] = int(s.nunique(dropna=True))
        r["top_share"] = float(vc.iloc[0]) if len(vc) else np.nan
    return r

all_rows = [row_for(c, "all", pd.NA, df_clean[c]) for c in features]
class_rows = []
for cls in classes:
    mask = y == cls
    for c in features:
        class_rows.append(row_for(c, "class", cls, df_clean.loc[mask, c]))

rows = all_rows + class_rows
if len(rows) > 50:
    keep_all = min(len(all_rows), 50)
    rows = all_rows[:keep_all] + class_rows[:max(0, 50 - keep_all)]

summary = pd.DataFrame(rows)

def round_sig(v):
    if pd.isna(v):
        return v
    v = float(v)
    if v == 0:
        return 0.0
    return round(v, -int(math.floor(math.log10(abs(v)))) + 3)

for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[col] = summary[col].map(round_sig)

mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distributions by target class")
n_numeric = int(summary.loc[summary["kind"] == "numeric", "variable"].nunique())
n_categorical = int(summary.loc[summary["kind"] == "categorical", "variable"].nunique())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"rows={len(df)} features={len(features)} numeric={n_numeric} categorical={n_categorical}")
print(f"placeholder tokens in {placeholder_cols} text feature columns")
print(f"distributions rows={len(summary)}")

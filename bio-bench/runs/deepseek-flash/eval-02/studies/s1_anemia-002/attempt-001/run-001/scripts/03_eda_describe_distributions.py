import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

missing = [k for k in ["targets", "excluded_columns", "group_column", "task"] if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
task = protocol["task"]
target_col = targets[0]

df = pd.read_parquet(paths["anemia"])
if target_col not in df.columns:
    raise KeyError(f"Target {target_col} not in data columns {list(df.columns)}")

def is_key_like(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.notna().all() and s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.notna().all() and s.nunique(dropna=True) == len(s)
    return False

excl = set(targets) | set(excluded)
if group_col:
    excl.add(group_col)
feature_cols = [c for c in df.columns if c not in excl and not is_key_like(df[c])]

numeric_cols, categorical_cols = [], []
for c in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(df[c]):
        new = c + "_days"
        df[new] = (df[c] - df[c].min()).dt.days
        numeric_cols.append(new)
    elif pd.api.types.is_numeric_dtype(df[c]):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

placeholders = {"?", "-", "NA", "N/A"}
ph_cols = []
for c in categorical_cols:
    s = df[c].astype("string").str.strip()
    if s.isin(list(placeholders)).any() or s.eq("").any():
        ph_cols.append(c)

def sig4(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        return float(f"{x:.4g}")
    except Exception:
        return x

def make_row(feature, typ, group, s):
    if typ == "numeric":
        s2 = pd.to_numeric(s, errors="coerce").dropna()
        n = len(s2)
        n_missing = len(s) - n
        if n:
            mean, sd = s2.mean(), (s2.std(ddof=1) if n > 1 else np.nan)
            mn, med, mx = s2.min(), s2.median(), s2.max()
        else:
            mean = sd = mn = med = mx = np.nan
        return dict(feature=feature, type=typ, group=group, n=n, n_missing=n_missing,
                    mean=mean, sd=sd, min=mn, median=med, max=mx,
                    n_levels=np.nan, top_share=np.nan)
    s2 = s.dropna()
    n = len(s2)
    n_missing = len(s) - n
    if n:
        vc = s2.value_counts()
        n_levels = s2.nunique()
        top_share = vc.iloc[0] / n if len(vc) else np.nan
    else:
        n_levels, top_share = 0, np.nan
    return dict(feature=feature, type=typ, group=group, n=n, n_missing=n_missing,
                mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                n_levels=n_levels, top_share=top_share)

rows = []
for c in numeric_cols:
    rows.append(make_row(c, "numeric", "overall", df[c]))
for c in categorical_cols:
    rows.append(make_row(c, "categorical", "overall", df[c]))

classes = list(df[target_col].dropna().unique())
for c in numeric_cols + categorical_cols:
    typ = "numeric" if c in numeric_cols else "categorical"
    for cls in classes:
        if len(rows) >= 50:
            break
        rows.append(make_row(c, typ, str(cls), df.loc[df[target_col] == cls, c]))
    if len(rows) >= 50:
        break

out = pd.DataFrame(rows[:50], columns=[
    "feature", "type", "group", "n", "n_missing", "mean", "sd",
    "min", "median", "max", "n_levels", "top_share"
])
for c in ["n", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_share"]:
    out[c] = out[c].map(sig4)

mlca_runtime.save_table(out, "distributions.csv", caption="Column distribution summaries")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")

print(f"distributions rows={len(out)} numeric={len(numeric_cols)} categorical={len(categorical_cols)}")
print(f"placeholder tokens ?, -, NA, N/A, whitespace in {len(ph_cols)} text columns")
print("saved distributions.csv")

import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column")
task = protocol.get("task", {})

df = pd.read_parquet(inputs["heart_disease_cleveland_uci"])

# placeholder tokens -> NaN in object columns (RL-0023)
placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = 0
for c in df.columns:
    if df[c].dtype == object:
        s = df[c].astype(str).str.strip()
        mask = s.isin(placeholders) | (s == "")
        if mask.any():
            ph_cols += 1
            df[c] = df[c].where(~mask, np.nan)
        else:
            df[c] = df[c].where(s != "", np.nan)
print(f"placeholder columns: {ph_cols}")

drop_cols = set(targets) | set(excluded)
if group_col:
    drop_cols.add(group_col)
feat_cols = [c for c in df.columns if c not in drop_cols]

# drop key-like columns (RL-0007)
key_cols = []
for c in feat_cols:
    s = df[c]
    non_na = s.dropna()
    if non_na.empty:
        continue
    if (pd.api.types.is_integer_dtype(s.dtype) or s.dtype == object) and non_na.is_unique and len(non_na) == len(df):
        key_cols.append(c)
feat_cols = [c for c in feat_cols if c not in key_cols]

numeric_cols, cat_cols = [], []
for c in feat_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
        numeric_cols.append(c)
    elif pd.api.types.is_datetime64_any_dtype(s.dtype):
        continue  # skip raw datetimes in summaries
    else:
        cat_cols.append(c)

def summarize(s, kind):
    if kind == "numeric":
        v = pd.to_numeric(s, errors="coerce")
        return dict(n=int(v.notna().sum()), n_missing=int(v.isna().sum()),
                    mean=v.mean(), sd=v.std(), min=v.min(), median=v.median(),
                    max=v.max(), n_levels=np.nan, top_share=np.nan)
    v = s.astype(object)
    n = int(v.notna().sum())
    vc = v.value_counts(dropna=True)
    nl = int(vc.shape[0])
    top = float(vc.iloc[0] / n) if n > 0 and nl > 0 else np.nan
    return dict(n=n, n_missing=int(v.isna().sum()),
                mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                n_levels=nl, top_share=top)

rows = []
for c in numeric_cols:
    rows.append(dict(variable=c, type="numeric", group="all", **summarize(df[c], "numeric")))
for c in cat_cols:
    rows.append(dict(variable=c, type="categorical", group="all", **summarize(df[c], "categorical")))

if len(targets) == 1 and targets[0] in df.columns and task.get("type") in ("binary", "multiclass"):
    tgt = targets[0]
    for cls in sorted(df[tgt].dropna().unique()):
        mask = df[tgt] == cls
        for c in numeric_cols:
            rows.append(dict(variable=c, type="numeric", group=f"{tgt}={cls}",
                             **summarize(df.loc[mask, c], "numeric")))

cols = ["variable", "type", "group", "n", "n_missing", "mean", "sd", "min",
        "median", "max", "n_levels", "top_share"]
out = pd.DataFrame(rows, columns=cols).head(50)
for k in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[k] = out[k].apply(lambda v: round(float(v), 4) if pd.notna(v) else np.nan)

mlca_runtime.save_table(out, "distributions.csv", caption="Feature distributions overall and by target class")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")
print(f"rows={out.shape[0]} numeric={len(numeric_cols)} categorical={len(cat_cols)} key_dropped={len(key_cols)}")

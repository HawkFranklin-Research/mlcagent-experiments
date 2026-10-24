import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
df = pd.read_parquet(inp["rna_er"])
protocol = json.loads(inp["_protocol"].read_text())

targets = protocol["targets"]
if isinstance(targets, str):
    targets = [targets]
excluded = protocol["excluded_columns"]
if isinstance(excluded, dict):
    excluded = excluded.get("columns", [])
group_col = protocol.get("group_column")

all_cols = list(df.columns)
feat = [c for c in all_cols if c not in set(targets) and c not in set(excluded) and c != group_col]

n = len(df)
drop_keys = []
for c in feat:
    s = df[c]
    nonnull = int(s.notna().sum())
    if (s.dtype == object or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == nonnull and nonnull > 1:
        drop_keys.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == nonnull and nonnull > 1:
        if any(k in c.lower() for k in ["id", "key", "index"]):
            drop_keys.append(c)
feat = [c for c in feat if c not in set(drop_keys)]

found = set()
placeholder_cols = 0
for c in feat:
    s = df[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        stripped = s.astype("string").str.strip()
        mask = stripped.isin(["?", "-", "NA", "N/A", ""])
        if mask.any():
            placeholder_cols += 1
            found.update(stripped[mask].dropna().unique().tolist())
            df[c] = stripped.mask(mask, pd.NA)
print(f"placeholders {sorted(found)} in {placeholder_cols} columns")

numeric_cols, cat_cols, dt_cols = [], [], []
for c in feat:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif pd.api.types.is_datetime64_any_dtype(s):
        dt_cols.append(c)
    else:
        try:
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().mean() > 0.9 and s.notna().any():
                dt_cols.append(c)
                continue
        except Exception:
            pass
        cat_cols.append(c)

n_numeric = len(numeric_cols)
n_categorical = len(cat_cols)
print(f"features: numeric={n_numeric}, categorical={n_categorical}, datetime={len(dt_cols)}")

def num_stats(s, var, tgt):
    s = pd.to_numeric(s, errors="coerce")
    return dict(variable=var, type="numeric", target_class=tgt, n=int(s.notna().sum()),
                n_missing=int(s.isna().sum()), mean=s.mean(), sd=s.std(ddof=1),
                min=s.min(), median=s.median(), max=s.max(), n_levels=np.nan, top_share=np.nan)

def cat_stats(s, var, tgt, all_s):
    s = s.dropna()
    vc = s.value_counts(normalize=True)
    return dict(variable=var, type="categorical", target_class=tgt, n=int(s.shape[0]),
                n_missing=int(all_s.isna().sum()), mean=np.nan, sd=np.nan,
                min=np.nan, median=np.nan, max=np.nan,
                n_levels=int(s.nunique()), top_share=float(vc.iloc[0]) if len(vc) else np.nan)

num_sel = numeric_cols[:20]
cat_sel = cat_cols[:10]
rows = []
for c in num_sel:
    rows.append(num_stats(df[c], c, "all"))
for c in cat_sel:
    rows.append(cat_stats(df[c], c, "all", df[c]))

tgt = targets[0] if targets else None
if tgt and tgt in df.columns:
    classes = df[tgt].dropna().unique().tolist()
    if 1 < len(classes) <= 5:
        for c in num_sel[:10]:
            for cls in classes:
                rows.append(num_stats(df.loc[df[tgt] == cls, c], c, str(cls)))

summary = pd.DataFrame(rows).head(50)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    if c in summary.columns:
        summary[c] = summary[c].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)

mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distribution summary")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"saved rows={len(summary)}")

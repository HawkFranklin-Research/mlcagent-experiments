import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
df = pd.read_parquet(inputs['rna_her2'])

try:
    targets = protocol['targets']
    excluded = protocol['excluded_columns']
    if 'group_column' not in protocol:
        raise KeyError(f"Missing 'group_column'. Available keys: {list(protocol.keys())}")
    group_col = protocol['group_column']
    task = protocol['task']
except KeyError as e:
    raise KeyError(f"Missing protocol key: {e}. Available keys: {list(protocol.keys())}")

target = targets[0] if targets else None
if target is None:
    raise ValueError("No target declared in protocol['targets']")

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]
nrows = len(df)
keep = []
for c in feature_cols:
    s = df[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique() == nrows:
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique() == nrows:
        if 'id' in c.lower() or 'code' in c.lower() or 'key' in c.lower():
            continue
    keep.append(c)
feature_cols = keep

df_feat = df[feature_cols].copy()
placeholders = {'?', '-', 'NA', 'N/A', ''}
placeholder_cols = 0
for c in df_feat.columns:
    if pd.api.types.is_object_dtype(df_feat[c]) or pd.api.types.is_string_dtype(df_feat[c]):
        s = df_feat[c].astype(str).str.strip()
        mask = s.isin(placeholders) | (s == '')
        if mask.any():
            placeholder_cols += 1
            df_feat[c] = s.where(~mask, np.nan)
        else:
            df_feat[c] = s
if placeholder_cols:
    print(f"Placeholders found in {placeholder_cols} text columns")

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_feat[c]) and not pd.api.types.is_bool_dtype(df_feat[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

n_classes = df[target].nunique()
groups = [('all', pd.Series(True, index=df.index))]
if n_classes <= 5:
    for val in df[target].dropna().unique():
        groups.append((str(val), df[target] == val))
max_feat = 50 // len(groups) if groups else 50
selected = (numeric_cols[:8] + categorical_cols[:8])[:max_feat]

def summarize(col, subset, group_label):
    vals = subset[col].dropna()
    n = len(vals)
    n_missing = subset[col].isna().sum()
    if col in numeric_cols:
        return dict(variable=col, type='numeric', group=group_label, n=n,
                    mean=vals.mean() if n else np.nan,
                    sd=vals.std() if n else np.nan,
                    min=vals.min() if n else np.nan,
                    median=vals.median() if n else np.nan,
                    max=vals.max() if n else np.nan,
                    n_missing=n_missing, n_levels=np.nan, top_share=np.nan)
    else:
        n_levels = vals.nunique() if n else np.nan
        top_share = (vals.value_counts().iloc[0] / n) if n else np.nan
        return dict(variable=col, type='categorical', group=group_label, n=n,
                    mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                    n_missing=n_missing, n_levels=n_levels, top_share=top_share)

rows = []
for col in selected:
    for label, mask in groups:
        subset = df_feat[mask]
        rows.append(summarize(col, subset, label))

out = pd.DataFrame(rows)
out = out[['variable','type','group','n','mean','sd','min','median','max','n_missing','n_levels','top_share']]
out = out.round(4)
mlca_runtime.save_table(out, "distributions.csv", caption="Feature distribution summary")

mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")

print(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
print(f"Saved distributions.csv with {len(out)} rows")

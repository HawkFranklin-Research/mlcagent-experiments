import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
for k in required:
    if k not in protocol:
        raise KeyError(f"Missing key {k} in protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']
task = protocol['task']

df = pd.read_parquet(inputs['dry_bean_uci'])

drop_cols = set(targets) | set(excluded)
if group_col is not None:
    drop_cols.add(group_col)
feature_cols = [c for c in df.columns if c not in drop_cols]

key_like = []
for c in feature_cols:
    if df[c].dtype == 'object' or df[c].dtype.name == 'string':
        if df[c].nunique() == len(df):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(df[c]):
        if df[c].nunique() == len(df):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

text_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype.name == 'string']
placeholders = ['?', '-', 'NA', 'N/A', '']
found = set()
cols_with = 0
for c in text_cols:
    s = df[c].astype(str).str.strip()
    for p in placeholders:
        if (s == p).any():
            found.add(p)
            cols_with += 1
            break
if found:
    print(f"Placeholders found: {found} in {cols_with} columns")
else:
    print("No placeholder tokens found in text columns.")

rows = []
for c in feature_cols:
    s = df[c]
    n = s.notna().sum()
    n_missing = s.isna().sum()
    if pd.api.types.is_numeric_dtype(s):
        mean = s.mean()
        sd = s.std()
        minv = s.min()
        median = s.median()
        maxv = s.max()
        n_levels = np.nan
        top_share = np.nan
        typ = 'numeric'
    else:
        mean = sd = minv = median = maxv = np.nan
        n_levels = s.nunique()
        vc = s.value_counts(normalize=True)
        top_share = vc.iloc[0] if len(vc) > 0 else np.nan
        typ = 'categorical'
    rows.append({
        'group': 'overall', 'variable': c, 'type': typ,
        'n': n, 'n_missing': n_missing,
        'mean': mean, 'sd': sd, 'min': minv, 'median': median, 'max': maxv,
        'n_levels': n_levels, 'top_share': top_share
    })

max_rows = 50
overall_rows = len(feature_cols)
remaining = max_rows - overall_rows
numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
if remaining > 0 and len(numeric_cols) > 0:
    n_classes = df[targets[0]].nunique()
    max_features = remaining // n_classes
    if max_features > 0:
        variances = df[numeric_cols].var()
        top_numeric = variances.nlargest(min(max_features, len(numeric_cols))).index.tolist()
        for cls in df[targets[0]].unique():
            sub = df[df[targets[0]] == cls]
            for c in top_numeric:
                s = sub[c]
                n = s.notna().sum()
                n_missing = s.isna().sum()
                mean = s.mean()
                sd = s.std()
                minv = s.min()
                median = s.median()
                maxv = s.max()
                rows.append({
                    'group': f'class: {cls}', 'variable': c, 'type': 'numeric',
                    'n': n, 'n_missing': n_missing,
                    'mean': mean, 'sd': sd, 'min': minv, 'median': median, 'max': maxv,
                    'n_levels': np.nan, 'top_share': np.nan
                })

dist_df = pd.DataFrame(rows)
for col in ['mean', 'sd', 'min', 'median', 'max', 'top_share']:
    dist_df[col] = dist_df[col].apply(lambda x: float(f'{x:.4g}') if pd.notna(x) else x)

mlca_runtime.save_table(dist_df, "distributions.csv", caption="Distribution summary")

n_numeric = sum(1 for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]))
n_categorical = sum(1 for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c]))
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"Features: {len(feature_cols)} (numeric={n_numeric}, categorical={n_categorical})")
print(f"Summary rows: {len(dist_df)}")
print("Saved distributions.csv")

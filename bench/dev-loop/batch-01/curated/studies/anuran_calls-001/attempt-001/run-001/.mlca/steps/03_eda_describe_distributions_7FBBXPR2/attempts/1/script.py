import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
for k in required:
    if k not in protocol:
        raise KeyError(f"Missing key '{k}' in _protocol. Available keys: {list(protocol.keys())}")

df = pd.read_parquet(inputs['anuran_calls_uci'])

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

if group_col not in df.columns:
    raise ValueError(f"Group column '{group_col}' not in dataframe columns: {df.columns.tolist()}")

def is_key_like(s):
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s):
            return s.nunique() == len(s) and len(s) > 1
        return False
    else:
        return s.nunique() == len(s) and len(s) > 1

all_cols = df.columns.tolist()
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]
feature_cols = [c for c in feature_cols if not is_key_like(df[c])]

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

rows = []
for col in numeric_cols:
    s = df[col]
    rows.append({
        'variable': col, 'type': 'numeric',
        'n_missing': int(s.isna().sum()),
        'mean': s.mean(), 'sd': s.std(), 'min': s.min(),
        'median': s.median(), 'max': s.max(),
        'n_levels': np.nan, 'top_share': np.nan,
        'n': np.nan, 'share': np.nan
    })

for col in categorical_cols:
    s = df[col]
    vc = s.value_counts(dropna=True)
    n_levels = s.nunique(dropna=True)
    top_share = vc.iloc[0] / len(s) if n_levels > 0 else np.nan
    rows.append({
        'variable': col, 'type': 'categorical',
        'n_missing': int(s.isna().sum()),
        'mean': np.nan, 'sd': np.nan, 'min': np.nan,
        'median': np.nan, 'max': np.nan,
        'n_levels': n_levels, 'top_share': top_share,
        'n': np.nan, 'share': np.nan
    })

for target in targets:
    if target not in df.columns:
        continue
    vc = df[target].value_counts(dropna=True)
    for level, count in vc.head(5).items():
        rows.append({
            'variable': f"{target}:{level}", 'type': 'target_class',
            'n_missing': np.nan, 'mean': np.nan, 'sd': np.nan,
            'min': np.nan, 'median': np.nan, 'max': np.nan,
            'n_levels': np.nan, 'top_share': np.nan,
            'n': int(count), 'share': count / len(df)
        })

df_summary = pd.DataFrame(rows)

def round_sig(x, sig=4):
    if pd.isna(x):
        return x
    if x == 0:
        return 0.0
    return float(f"{x:.{sig}g}")

for col in ['mean', 'sd', 'min', 'median', 'max', 'top_share', 'share']:
    if col in df_summary.columns:
        df_summary[col] = df_summary[col].apply(round_sig)

mlca_runtime.save_table(df_summary, "distributions.csv", caption="Summary distributions of numeric and categorical features and target classes.")

mlca_runtime.record_fact_candidate('n_numeric', len(numeric_cols), source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', len(categorical_cols), source='distributions.csv')

placeholders = {'?', '-', 'NA', 'N/A', ''}
found_placeholders = set()
cols_with_placeholders = 0
for col in categorical_cols:
    s = df[col].dropna().astype(str)
    stripped = s.str.strip()
    mask = stripped.isin(placeholders)
    if mask.any():
        cols_with_placeholders += 1
        found_placeholders.update(stripped[mask].unique())

if found_placeholders:
    print(f"Placeholders found: {sorted(found_placeholders)} in {cols_with_placeholders} columns")
else:
    print("No placeholders found in text columns")

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"Target classes summarized: {sum(1 for r in rows if r['type']=='target_class')}")
print(f"Saved distributions.csv with {len(df_summary)} rows")

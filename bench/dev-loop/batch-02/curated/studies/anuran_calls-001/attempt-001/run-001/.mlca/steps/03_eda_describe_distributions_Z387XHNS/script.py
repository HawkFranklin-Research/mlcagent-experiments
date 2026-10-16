import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing keys in _protocol: {missing}. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

df = pd.read_parquet(inputs["anuran_calls_uci"])

feature_cols = []
for col in df.columns:
    if col in targets or col in excluded or col == group_col:
        continue
    if df[col].dtype == 'object' or df[col].dtype == 'string':
        if df[col].nunique() == len(df):
            continue
    if df[col].dtype.kind in 'iu':
        if df[col].nunique() == len(df) and any(k in col.lower() for k in ['id', 'code', 'key']):
            continue
    feature_cols.append(col)

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

def round_sig(x, sig=4):
    if pd.isna(x) or x == 0:
        return x
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

rows = []
for col in numeric_cols:
    s = df[col]
    rows.append({
        'column': col, 'kind': 'numeric',
        'mean': s.mean(), 'sd': s.std(), 'min': s.min(),
        'median': s.median(), 'max': s.max(), 'n_missing': s.isna().sum(),
        'n_levels': np.nan, 'top_share': np.nan
    })
for col in categorical_cols:
    s = df[col]
    vc = s.value_counts(dropna=True)
    n_levels = s.nunique(dropna=True)
    top_share = vc.iloc[0] / len(s) if len(vc) > 0 else np.nan
    rows.append({
        'column': col, 'kind': 'categorical',
        'mean': np.nan, 'sd': np.nan, 'min': np.nan,
        'median': np.nan, 'max': np.nan, 'n_missing': s.isna().sum(),
        'n_levels': n_levels, 'top_share': top_share
    })

summary_df = pd.DataFrame(rows)
for c in ['mean', 'sd', 'min', 'median', 'max', 'top_share']:
    if c in summary_df.columns:
        summary_df[c] = summary_df[c].apply(lambda x: round_sig(x) if pd.notna(x) else x)

mlca_runtime.save_table(summary_df, "distributions.csv", caption="Distribution summary of feature columns")

n_numeric = len(numeric_cols)
n_categorical = len(categorical_cols)
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

placeholders = {'?', '-', 'NA', 'N/A', ''}
placeholder_cols = []
for col in categorical_cols:
    s = df[col].astype(str)
    mask = s.isin(placeholders) | s.str.strip().eq('')
    if mask.any():
        placeholder_cols.append(col)
if placeholder_cols:
    print(f"Placeholders found in {len(placeholder_cols)} columns: {placeholder_cols[:3]}")
else:
    print("No placeholder tokens found in text columns.")

print(f"Features: {n_numeric} numeric, {n_categorical} categorical")
print(f"Saved distributions.csv with {len(summary_df)} rows")

import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol.get('targets', [])
excluded = protocol.get('excluded_columns', [])

df = pd.read_parquet(inputs['dry_bean_uci'])
cols = [c for c in df.columns if c not in excluded]
num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in cols if c not in num_cols]
group_col = targets[0] if targets and targets[0] in df.columns else None

rows = []
for c in num_cols:
    s = df[c]
    rows.append({
        'variable': c, 'type': 'numeric', 'group': 'overall',
        'mean': s.mean(), 'sd': s.std(), 'min': s.min(),
        'median': s.median(), 'max': s.max(), 'n_missing': s.isna().sum(),
        'n_levels': np.nan, 'top_share': np.nan, 'count': s.notna().sum()
    })

if group_col:
    for c in num_cols[:3]:
        for cls, sub in df.groupby(group_col):
            s = sub[c]
            rows.append({
                'variable': c, 'type': 'numeric', 'group': str(cls),
                'mean': s.mean(), 'sd': s.std(), 'min': s.min(),
                'median': s.median(), 'max': s.max(), 'n_missing': s.isna().sum(),
                'n_levels': np.nan, 'top_share': np.nan, 'count': s.notna().sum()
            })

for c in cat_cols:
    s = df[c]
    vc = s.value_counts()
    rows.append({
        'variable': c, 'type': 'categorical', 'group': 'overall',
        'mean': np.nan, 'sd': np.nan, 'min': np.nan, 'median': np.nan,
        'max': np.nan, 'n_missing': s.isna().sum(),
        'n_levels': s.nunique(), 'top_share': vc.iloc[0] / len(s) if len(s) else np.nan,
        'count': len(s)
    })

if group_col:
    total = len(df)
    for cls, sub in df.groupby(group_col):
        cnt = len(sub)
        rows.append({
            'variable': group_col, 'type': 'categorical', 'group': str(cls),
            'mean': np.nan, 'sd': np.nan, 'min': np.nan, 'median': np.nan,
            'max': np.nan, 'n_missing': 0,
            'n_levels': np.nan, 'top_share': cnt / total, 'count': cnt
        })

summary = pd.DataFrame(rows)
for col in ['mean', 'sd', 'min', 'median', 'max', 'n_missing', 'n_levels', 'top_share', 'count']:
    if col in summary.columns:
        summary[col] = summary[col].apply(lambda x: float(f'{x:.4g}') if pd.notna(x) else x)

mlca_runtime.save_table(summary, "distributions.csv", caption="Summary statistics of numeric and categorical columns")

n_numeric = summary[summary['type'] == 'numeric']['variable'].nunique()
n_categorical = summary[summary['type'] == 'categorical']['variable'].nunique()
mlca_runtime.record_fact_candidate('n_numeric', int(n_numeric), source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', int(n_categorical), source='distributions.csv')

print(f"Rows in summary: {len(summary)}")
print(f"Numeric columns: {n_numeric}, Categorical columns: {n_categorical}")

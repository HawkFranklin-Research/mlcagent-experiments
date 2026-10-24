import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
df = pd.read_parquet(inputs['discovery'])

targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)
task = protocol['task']

feat = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]
key_like = []
for c in feat:
    s = df[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_integer_dtype(s)) and not pd.api.types.is_bool_dtype(s):
        if s.nunique(dropna=True) == len(s):
            key_like.append(c)
feat = [c for c in feat if c not in key_like]

placeholder_cols = 0
for c in feat:
    if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
        vals = df[c].dropna().astype(str).str.strip()
        if vals.isin(['?', '-', 'NA', 'N/A', '']).any():
            placeholder_cols += 1
print(f"placeholder columns: {placeholder_cols}")

numeric_cols = [c for c in feat if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
categorical_cols = [c for c in feat if c not in numeric_cols]

target = targets[0]
groups = ['overall'] + sorted(df[target].dropna().unique().tolist())

rows = []
for g in groups:
    sub = df if g == 'overall' else df[df[target] == g]
    if numeric_cols:
        stats = []
        for c in numeric_cols:
            s = sub[c]
            stats.append({
                'mean': s.mean(), 'sd': s.std(), 'min': s.min(),
                'median': s.median(), 'max': s.max(), 'n_missing': s.isna().sum()
            })
        st = pd.DataFrame(stats)
        rows.append({
            'group': g, 'kind': 'numeric', 'n_columns': len(numeric_cols),
            'mean_of_means': st['mean'].mean(), 'sd_of_means': st['mean'].std(),
            'min_of_mins': st['min'].min(), 'median_of_medians': st['median'].median(),
            'max_of_maxes': st['max'].max(), 'mean_n_missing': st['n_missing'].mean(),
            'mean_n_levels': np.nan, 'mean_top_share': np.nan
        })
    if categorical_cols:
        stats = []
        for c in categorical_cols:
            s = sub[c]
            vc = s.value_counts(normalize=True, dropna=True)
            top = vc.iloc[0] if len(vc) > 0 else np.nan
            stats.append({'n_levels': s.nunique(dropna=True), 'top_share': top, 'n_missing': s.isna().sum()})
        st = pd.DataFrame(stats)
        rows.append({
            'group': g, 'kind': 'categorical', 'n_columns': len(categorical_cols),
            'mean_of_means': np.nan, 'sd_of_means': np.nan,
            'min_of_mins': np.nan, 'median_of_medians': np.nan,
            'max_of_maxes': np.nan, 'mean_n_missing': st['n_missing'].mean(),
            'mean_n_levels': st['n_levels'].mean(), 'mean_top_share': st['top_share'].mean()
        })

out = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(out, "distributions.csv", caption="Summary of numeric and categorical feature columns by target class")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")
print(f"features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
print(f"saved distributions.csv with {len(out)} rows")

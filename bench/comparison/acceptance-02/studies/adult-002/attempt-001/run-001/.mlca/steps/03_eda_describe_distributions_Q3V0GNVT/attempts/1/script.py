import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
df = pd.read_parquet(inputs['adult_census_uci'])

if 'targets' not in protocol:
    raise KeyError(f"Missing 'targets' in protocol. Available keys: {list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
if 'excluded_columns' not in protocol:
    raise KeyError(f"Missing 'excluded_columns' in protocol. Available keys: {list(protocol.keys())}")
group_col = protocol.get('group_column', None)

placeholder_tokens = ['?', '-', 'NA', 'N/A', '']
placeholder_cols = []
for col in df.columns:
    if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
        series = df[col].astype(str).str.strip()
        has_placeholder = any((series == tok).any() for tok in placeholder_tokens)
        if has_placeholder:
            placeholder_cols.append(col)
if placeholder_cols:
    print(f"Placeholder tokens found in {len(placeholder_cols)} columns: {placeholder_cols}")
else:
    print("No placeholder tokens found.")

base_exclude = set(targets) | set(excluded)
if group_col:
    base_exclude.add(group_col)
feature_cols = [c for c in df.columns if c not in base_exclude]

numeric_cols = []
categorical_cols = []
datetime_cols = []
for col in feature_cols:
    s = df[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        datetime_cols.append(col)
        continue
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=False) == len(s):
            continue
        categorical_cols.append(col)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(s):
            vals = np.sort(s.dropna().unique())
            if len(vals) == len(s) and (vals[0] in (0, 1)) and (vals[-1] in (len(s)-1, len(s))):
                if np.all(np.diff(vals) == 1):
                    continue
        numeric_cols.append(col)
    elif pd.api.types.is_float_dtype(s) or pd.api.types.is_bool_dtype(s):
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

df_feat = df.copy()
for col in datetime_cols:
    dt = pd.to_datetime(df_feat[col], errors='coerce')
    for part, func in [('year', dt.dt.year), ('month', dt.dt.month), ('day', dt.dt.day),
                       ('dow', dt.dt.dayofweek), ('hour', dt.dt.hour)]:
        new_col = f"{col}_{part}"
        df_feat[new_col] = func
        numeric_cols.append(new_col)

rows = []
def add_rows(class_label, frame):
    for col in numeric_cols:
        s = frame[col]
        rows.append({
            'variable': col, 'class': class_label, 'type': 'numeric',
            'mean': s.mean(), 'sd': s.std(), 'min': s.min(), 'median': s.median(),
            'max': s.max(), 'n_missing': s.isna().sum(),
            'n_levels': np.nan, 'top_share': np.nan,
        })
    for col in categorical_cols:
        s = frame[col]
        nlev = s.nunique(dropna=True)
        top = s.value_counts(normalize=True, dropna=True).iloc[0] if nlev > 0 else np.nan
        rows.append({
            'variable': col, 'class': class_label, 'type': 'categorical',
            'mean': np.nan, 'sd': np.nan, 'min': np.nan, 'median': np.nan,
            'max': np.nan, 'n_missing': s.isna().sum(),
            'n_levels': nlev, 'top_share': top,
        })

add_rows('overall', df_feat)

if len(targets) == 1:
    target_col = targets[0]
    if target_col in df.columns:
        for cls in df[target_col].dropna().unique():
            mask = df[target_col] == cls
            add_rows(f"{target_col}={cls}", df_feat[mask])

dist_df = pd.DataFrame(rows)
for col in ['mean', 'sd', 'min', 'median', 'max', 'n_missing', 'n_levels', 'top_share']:
    if col in dist_df.columns:
        dist_df[col] = dist_df[col].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) and isinstance(x, (int, float)) else x)
if len(dist_df) > 50:
    dist_df = dist_df.head(50)

mlca_runtime.save_table(dist_df, "distributions.csv", caption="Distribution summary by variable and target class")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")
print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Summary rows: {len(dist_df)}")

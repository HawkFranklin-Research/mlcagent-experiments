import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["breast_cancer_diagnostic_uci"])

targets = proto.get("targets")
if targets is None:
    raise KeyError(f"Missing 'targets' in protocol. Available keys: {list(proto.keys())}")
excluded = proto.get("excluded_columns", [])
group = proto.get("group_column", None)
task = proto.get("task", {})

all_cols = list(df.columns)
drop = set(targets) | set(excluded)
if group:
    drop.add(group)
candidate = [c for c in all_cols if c not in drop]

feature_cols = []
for c in candidate:
    s = df[c]
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=False) == len(df):
            continue
    feature_cols.append(c)

placeholders = {'?', '-', 'NA', 'N/A'}
found_placeholders = set()
placeholder_cols = []
for c in feature_cols:
    if not pd.api.types.is_numeric_dtype(df[c]):
        vals = df[c].astype(str).str.strip()
        mask = vals.isin(placeholders) | (vals == '')
        if mask.any():
            placeholder_cols.append(c)
            found_placeholders.update(vals[mask].unique())
print(f"Placeholders {found_placeholders} in {len(placeholder_cols)} columns")

rows = []
for c in feature_cols:
    s = df[c]
    is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    row = {'variable': c, 'type': 'numeric' if is_num else 'categorical', 'group': 'overall'}
    if is_num:
        row.update({
            'mean': s.mean(), 'sd': s.std(), 'min': s.min(),
            'median': s.median(), 'max': s.max(), 'n_missing': s.isna().sum(),
            'n_levels': np.nan, 'top_share': np.nan
        })
    else:
        vc = s.value_counts(dropna=True)
        top = vc.iloc[0] if len(vc) > 0 else np.nan
        row.update({
            'mean': np.nan, 'sd': np.nan, 'min': np.nan,
            'median': np.nan, 'max': np.nan, 'n_missing': s.isna().sum(),
            'n_levels': s.nunique(dropna=True),
            'top_share': top / len(s) if len(s) > 0 else np.nan
        })
    rows.append(row)

for t in targets:
    s = df[t]
    vc = s.value_counts(dropna=True)
    top = vc.iloc[0] if len(vc) > 0 else np.nan
    rows.append({
        'variable': t, 'type': 'categorical', 'group': 'overall',
        'mean': np.nan, 'sd': np.nan, 'min': np.nan, 'median': np.nan, 'max': np.nan,
        'n_missing': s.isna().sum(), 'n_levels': s.nunique(dropna=True),
        'top_share': top / len(s) if len(s) > 0 else np.nan
    })
    for cls in s.dropna().unique():
        rows.append({
            'variable': t, 'type': 'categorical', 'group': f'{t}={cls}',
            'mean': np.nan, 'sd': np.nan, 'min': np.nan, 'median': np.nan, 'max': np.nan,
            'n_missing': 0, 'n_levels': 1, 'top_share': 1.0
        })

df_sum = pd.DataFrame(rows)
cols = ['variable', 'type', 'group', 'mean', 'sd', 'min', 'median', 'max', 'n_missing', 'n_levels', 'top_share']
df_sum = df_sum[cols]

def sig_round(x, sig=4):
    if pd.isna(x) or x == 0:
        return x
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

for c in ['mean', 'sd', 'min', 'median', 'max', 'top_share']:
    df_sum[c] = df_sum[c].apply(lambda x: sig_round(x) if pd.notna(x) else np.nan)

df_sum['n_missing'] = df_sum['n_missing'].astype('Int64')
df_sum['n_levels'] = df_sum['n_levels'].astype('Int64')

mlca_runtime.save_table(df_sum, "distributions.csv", caption="Distribution summary")

n_numeric = sum(1 for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]))
n_categorical = len(feature_cols) - n_numeric
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"n_numeric={n_numeric}, n_categorical={n_categorical}")
print(f"Saved distributions.csv with {len(df_sum)} rows")

import mlca_runtime
import pandas as pd
import numpy as np
import json
from math import floor, log10

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["heart_disease_cleveland_uci"])

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing keys: {missing}. Available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
task = protocol["task"]

all_cols = list(df.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

def is_key_like(series):
    if series.dtype == object or series.dtype == 'string':
        if series.nunique() == len(series) and not series.astype(str).str.match(r'^-?\d+(\.\d+)?$').all():
            return True
    return False

key_like = [c for c in feature_cols if is_key_like(df[c])]
feature_cols = [c for c in feature_cols if c not in key_like]

placeholders = {'?', '-', 'NA', 'N/A', ''}
placeholder_cols = set()
found_placeholders = set()
for c in feature_cols:
    if df[c].dtype == object or df[c].dtype == 'string':
        s = df[c].astype(str).str.strip()
        for ph in placeholders:
            if (s == ph).any():
                found_placeholders.add(ph)
                placeholder_cols.add(c)
        if (s == '').any():
            found_placeholders.add('whitespace')
            placeholder_cols.add(c)
        df[c] = s.replace(list(placeholders) + [''], pd.NA)

print(f"Placeholders found: {found_placeholders if found_placeholders else 'none'} in {len(placeholder_cols)} columns")

numeric_cols = []
categorical_cols = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 10:
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

rows = []
target_col = targets[0]

for c in feature_cols:
    if c in numeric_cols:
        s = df[c]
        rows.append({
            'variable': c, 'type': 'numeric', 'target_class': 'overall',
            'n_missing': s.isna().sum(),
            'mean': s.mean(), 'sd': s.std(), 'min': s.min(), 'median': s.median(), 'max': s.max(),
            'n_levels': np.nan, 'top_level_share': np.nan
        })
    else:
        s = df[c]
        vc = s.value_counts(dropna=True)
        rows.append({
            'variable': c, 'type': 'categorical', 'target_class': 'overall',
            'n_missing': s.isna().sum(),
            'mean': np.nan, 'sd': np.nan, 'min': np.nan, 'median': np.nan, 'max': np.nan,
            'n_levels': s.nunique(),
            'top_level_share': vc.iloc[0] / s.notna().sum() if len(vc) > 0 else np.nan
        })

for cls in df[target_col].dropna().unique():
    sub = df[df[target_col] == cls]
    for c in feature_cols:
        if c in numeric_cols:
            s = sub[c]
            rows.append({
                'variable': c, 'type': 'numeric', 'target_class': str(cls),
                'n_missing': s.isna().sum(),
                'mean': s.mean(), 'sd': s.std(), 'min': s.min(), 'median': s.median(), 'max': s.max(),
                'n_levels': np.nan, 'top_level_share': np.nan
            })
        else:
            s = sub[c]
            vc = s.value_counts(dropna=True)
            rows.append({
                'variable': c, 'type': 'categorical', 'target_class': str(cls),
                'n_missing': s.isna().sum(),
                'mean': np.nan, 'sd': np.nan, 'min': np.nan, 'median': np.nan, 'max': np.nan,
                'n_levels': s.nunique(),
                'top_level_share': vc.iloc[0] / s.notna().sum() if len(vc) > 0 else np.nan
            })

dist_df = pd.DataFrame(rows)

def round_sig(x, sig=4):
    if pd.isna(x) or x == 0:
        return x
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

for col in ['mean', 'sd', 'min', 'median', 'max', 'top_level_share']:
    dist_df[col] = dist_df[col].apply(lambda x: round_sig(x) if pd.notna(x) else x)

if len(dist_df) > 50:
    dist_df = dist_df.head(50)

mlca_runtime.save_table(dist_df, "distributions.csv", caption="Summary distributions")

n_numeric = len(numeric_cols)
n_categorical = len(categorical_cols)
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"Features: {len(feature_cols)} ({n_numeric} numeric, {n_categorical} categorical)")
print(f"Saved distributions.csv with {len(dist_df)} rows")

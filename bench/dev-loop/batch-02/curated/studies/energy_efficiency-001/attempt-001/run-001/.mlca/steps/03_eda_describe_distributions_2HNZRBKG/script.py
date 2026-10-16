import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def get_protocol_key(d, key):
    if key not in d:
        raise KeyError(f"Missing key '{key}' in _protocol. Available keys: {list(d.keys())}")
    return d[key]

targets = get_protocol_key(protocol, "targets")
excluded = get_protocol_key(protocol, "excluded_columns")
group_col = get_protocol_key(protocol, "group_column")
task = get_protocol_key(protocol, "task")

df = pd.read_parquet(inputs["energy_efficiency_uci"])

def is_text(s):
    return s.dtype == 'object' or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)

feature_cols = []
for col in df.columns:
    if col in targets or col in excluded or col == group_col:
        continue
    s = df[col]
    if is_text(s) and s.nunique(dropna=False) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        continue
    feature_cols.append(col)

rows = []
for col in feature_cols:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        kind = 'numeric'
        mean = s.mean()
        sd = s.std()
        min_val = s.min()
        median = s.median()
        max_val = s.max()
        n_missing = s.isna().sum()
        n_levels = np.nan
        top_share = np.nan
    else:
        kind = 'categorical'
        mean = sd = min_val = median = max_val = np.nan
        n_missing = np.nan
        n_levels = s.nunique(dropna=True)
        top_share = s.value_counts(normalize=True, dropna=True).iloc[0] if n_levels > 0 else np.nan
    rows.append({
        'variable': col,
        'kind': kind,
        'mean': mean,
        'sd': sd,
        'min': min_val,
        'median': median,
        'max': max_val,
        'n_missing': n_missing,
        'n_levels': n_levels,
        'top_share': top_share
    })

summary = pd.DataFrame(rows)

def round_sig(x, sig=4):
    if pd.isna(x) or x == 0:
        return x
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

for col in ['mean', 'sd', 'min', 'median', 'max', 'top_share']:
    summary[col] = summary[col].apply(lambda x: round_sig(x) if pd.notna(x) else x)

n_numeric = int((summary['kind'] == 'numeric').sum())
n_categorical = int((summary['kind'] == 'categorical').sum())

mlca_runtime.save_table(summary, "distributions.csv", caption="Summary distributions of feature columns")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"Features: {len(feature_cols)} columns, numeric={n_numeric}, categorical={n_categorical}")
print(f"Summary shape: {summary.shape}")

placeholders = ['?', '-', 'NA', 'N/A']
placeholder_cols = set()
found_placeholders = set()
for col in feature_cols:
    s = df[col]
    if is_text(s):
        s_str = s.dropna().astype(str)
        if (s_str != s_str.str.strip()).any():
            placeholder_cols.add(col)
            found_placeholders.add('whitespace')
        for p in placeholders:
            if (s_str == p).any():
                placeholder_cols.add(col)
                found_placeholders.add(p)
        if (s_str.str.strip() == '').any():
            placeholder_cols.add(col)
            found_placeholders.add('whitespace-only')
if placeholder_cols:
    print(f"Placeholders found: {', '.join(sorted(found_placeholders))} in {len(placeholder_cols)} columns")
else:
    print("No placeholder tokens found in text columns.")

import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["breast_cancer_diagnostic_uci"])

targets = protocol.get("targets")
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column", None)
task = protocol.get("task", {})

if targets is None:
    raise KeyError(f"Missing 'targets' in protocol. Available keys: {list(protocol.keys())}")

def is_key_like(col):
    if col in targets or col in excluded or col == group_col:
        return False
    s = df[col]
    if s.dtype == object or pd.api.types.is_integer_dtype(s):
        if s.nunique() == len(s):
            return True
    return False

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col and not is_key_like(c)]

placeholders = {'?', '-', 'NA', 'N/A'}
placeholder_cols = set()
for col in df.columns:
    if df[col].dtype == object:
        s = df[col].astype(str).str.strip()
        for ph in placeholders:
            if (s == ph).any():
                placeholder_cols.add(col)
                break
        if (s == '').any():
            placeholder_cols.add(col)
if placeholder_cols:
    print(f"Placeholders found in {len(placeholder_cols)} columns: {', '.join(sorted(placeholder_cols))}")
else:
    print("No placeholder tokens found in text columns.")

rows = []
numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

for col in feature_cols:
    s = df[col]
    n_missing = s.isna().sum()
    if col in numeric_cols:
        rows.append({
            'column': col,
            'kind': 'numeric',
            'n_missing': n_missing,
            'mean': s.mean(),
            'sd': s.std(),
            'min': s.min(),
            'median': s.median(),
            'max': s.max(),
            'n_levels': np.nan,
            'top_level': np.nan,
            'top_share': np.nan,
            'group': 'overall'
        })
    else:
        vc = s.value_counts(dropna=False)
        n_levels = s.nunique(dropna=False)
        top_level = vc.index[0] if len(vc) > 0 else np.nan
        top_share = vc.iloc[0] / len(s) if len(vc) > 0 else np.nan
        rows.append({
            'column': col,
            'kind': 'categorical',
            'n_missing': n_missing,
            'mean': np.nan,
            'sd': np.nan,
            'min': np.nan,
            'median': np.nan,
            'max': np.nan,
            'n_levels': n_levels,
            'top_level': top_level,
            'top_share': top_share,
            'group': 'overall'
        })

if targets:
    target_col = targets[0]
    if target_col in df.columns:
        classes = df[target_col].dropna().unique()
        n_classes = len(classes)
        max_rows = 50
        remaining = max_rows - len(rows)
        if remaining > 0 and n_classes > 0:
            n_break = min(len(numeric_cols), remaining // n_classes)
            break_cols = numeric_cols[:n_break]
            for cls in classes:
                sub = df[df[target_col] == cls]
                for col in break_cols:
                    s = sub[col]
                    rows.append({
                        'column': col,
                        'kind': 'numeric',
                        'n_missing': s.isna().sum(),
                        'mean': s.mean(),
                        'sd': s.std(),
                        'min': s.min(),
                        'median': s.median(),
                        'max': s.max(),
                        'n_levels': np.nan,
                        'top_level': np.nan,
                        'top_share': np.nan,
                        'group': f'class={cls}'
                    })

distributions_df = pd.DataFrame(rows)

def round_sig(x, sig=4):
    if pd.isna(x):
        return x
    if isinstance(x, (int, float, np.number)):
        return float(f"{x:.{sig}g}")
    return x

for col in ['n_missing', 'mean', 'sd', 'min', 'median', 'max', 'n_levels', 'top_share']:
    if col in distributions_df.columns:
        distributions_df[col] = distributions_df[col].apply(round_sig)

if len(distributions_df) > 50:
    distributions_df = distributions_df.iloc[:50]

mlca_runtime.save_table(distributions_df, "distributions.csv", caption="Summary distributions of feature columns")

n_numeric = len(numeric_cols)
n_categorical = len(categorical_cols)
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"Features: {len(feature_cols)} total, {n_numeric} numeric, {n_categorical} categorical")
print(f"Saved distributions.csv with {len(distributions_df)} rows")

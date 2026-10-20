import mlca_runtime, json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)

inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
df = pd.read_parquet(inp['bank_marketing_uci'])

for k in ['targets', 'excluded_columns', 'group_column', 'task']:
    if k not in proto:
        raise KeyError(f"Missing '{k}' in _protocol. Available top-level keys: {list(proto.keys())}")

targets = proto['targets']
excluded = proto['excluded_columns']
group = proto['group_column']
target = targets[0]

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feats = [c for c in df.columns if c not in drop]

def is_keylike(s):
    n = len(s)
    if n <= 1:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == n
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == n
    return False

feats = [c for c in feats if not is_keylike(df[c])]

num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
cat_cols = [c for c in feats if c not in num_cols]

n_numeric = len(num_cols)
n_categorical = len(cat_cols)

def summary_row(s, typ, cls):
    n_miss = int(s.isna().sum())
    if typ == 'numeric':
        n_ok = int(s.notna().sum())
        sd = float(s.std(ddof=1)) if n_ok > 1 else np.nan
        return {'variable': s.name, 'type': typ, 'target_class': cls, 'n_missing': n_miss,
                'mean': float(s.mean()), 'sd': sd, 'min': float(s.min()),
                'median': float(s.median()), 'max': float(s.max()),
                'n_levels': np.nan, 'top_share': np.nan}
    else:
        vc = s.value_counts(dropna=True)
        nlev = int(vc.size)
        top = float(vc.iloc[0] / vc.sum()) if nlev > 0 else np.nan
        return {'variable': s.name, 'type': typ, 'target_class': cls, 'n_missing': n_miss,
                'mean': np.nan, 'sd': np.nan, 'min': np.nan, 'median': np.nan, 'max': np.nan,
                'n_levels': nlev, 'top_share': top}

classes = sorted(df[target].dropna().unique())
rows = []
for cls in classes:
    sub = df[df[target] == cls]
    for c in num_cols:
        rows.append(summary_row(sub[c], 'numeric', cls))
    for c in cat_cols:
        rows.append(summary_row(sub[c], 'categorical', cls))
if len(rows) > 50:
    rows = []
    for c in num_cols:
        rows.append(summary_row(df[c], 'numeric', 'all'))
    for c in cat_cols:
        rows.append(summary_row(df[c], 'categorical', 'all'))

out = pd.DataFrame(rows)
for col in ['n_missing', 'mean', 'sd', 'min', 'median', 'max', 'n_levels', 'top_share']:
    if col in out.columns:
        out[col] = out[col].astype(float).round(4)

mlca_runtime.save_table(out, "distributions.csv", caption="Feature distributions by target class")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"rows={df.shape[0]} feats={len(feats)} num={n_numeric} cat={n_categorical} summary_rows={len(out)}")

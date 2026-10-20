import mlca_runtime
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(Path(inp['_protocol']).read_text())
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

for k in ['targets', 'excluded_columns', 'group_column', 'task']:
    if k not in protocol:
        raise KeyError(f"missing {k}; available: {sorted(protocol.keys())}")
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.dtype == object or str(s.dtype) == 'string':
        if s.nunique() == len(s) and len(s) > 0:
            continue
    feature_cols.append(c)

ph_hits = {}
for c in feature_cols:
    s = train[c]
    if s.dtype == object or str(s.dtype) == 'string':
        mask = s.astype(str).str.strip().isin(['?', '-', 'NA', 'N/A', ''])
        if mask.any():
            ph_hits[c] = int(mask.sum())
print(f"placeholders: {ph_hits if ph_hits else 'none'}")

numeric_cols, cat_cols, date_cols, high_card_cols = [], [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        n = s.nunique(dropna=True)
        if n > 2 and n <= max(20, int(0.05 * len(train))) and s.dropna().astype(float).apply(float.is_integer).all():
            cat_cols.append(c)
        else:
            numeric_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        n = s.nunique(dropna=True)
        if n <= max(1, int(0.05 * len(train))):
            cat_cols.append(c)
        else:
            high_card_cols.append(c)
    else:
        numeric_cols.append(c)

feature_cols = numeric_cols + cat_cols + date_cols
print(f"features: {len(feature_cols)} (num={len(numeric_cols)}, cat={len(cat_cols)}, date={len(date_cols)}, high_card_dropped={len(high_card_cols)})")

ct_parts = []
for c in numeric_cols:
    s = train[c]
    steps = []
    if s.min() >= 0 and s.nunique() > 5:
        steps.append(('log', FunctionTransformer(np.log1p, validate=False, feature_names_out='one-to-one')))
    steps.append(('imp', SimpleImputer(strategy='median')))
    steps.append(('sc', StandardScaler()))
    ct_parts.append((c, Pipeline(steps), [c]))

if cat_cols:
    ct_parts.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
if date_cols:
    raise NotImplementedError("date handling not implemented")

ct = ColumnTransformer(ct_parts, remainder='drop', verbose_feature_names_out=False)
ct.set_output(transform='pandas')

X_train = train[feature_cols].copy()
ct = mlca_runtime.fit_record(ct, X_train, 'train')

mlca_runtime.save_model(ct, 'preprocessor.pkl')
print("preprocessor.pkl saved")

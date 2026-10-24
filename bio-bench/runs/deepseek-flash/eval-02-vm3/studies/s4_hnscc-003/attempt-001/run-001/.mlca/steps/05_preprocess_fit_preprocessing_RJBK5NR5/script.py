import mlca_runtime
import pandas as pd
import numpy as np
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]

key_cols = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        try:
            has_digit = s.astype(str).str.contains(r'\d').any()
        except Exception:
            has_digit = True
        if s.nunique() == len(s) and not has_digit:
            key_cols.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique() == len(s) and s.nunique() > 10:
        if (s.max() - s.min()) > 1e6:
            key_cols.append(c)
for c in key_cols:
    print(f"Excluding key-like column: {c}")
feature_cols = [c for c in feature_cols if c not in key_cols]

placeholder_tokens = ['?', '-', 'NA', 'N/A']
text_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c])]
ph_cols = [c for c in text_cols if train[c].astype(str).str.strip().isin(placeholder_tokens).any()]
if ph_cols:
    print(f"Placeholders found in {len(ph_cols)} columns")
    for c in ph_cols:
        train[c] = train[c].astype(str).str.strip().replace(placeholder_tokens, np.nan)
        val[c] = val[c].astype(str).str.strip().replace(placeholder_tokens, np.nan)

for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_float_dtype(train[c]):
        train[c] = train[c].replace([np.inf, -np.inf], np.nan)
        if c in val.columns:
            val[c] = val[c].replace([np.inf, -np.inf], np.nan)

excluded_features = []
for target in targets:
    if target not in train.columns or not pd.api.types.is_numeric_dtype(train[target]):
        continue
    t_num = train[target].astype(float)
    for c in feature_cols:
        if c in excluded_features or not pd.api.types.is_numeric_dtype(train[c]):
            continue
        v = train[c].astype(float)
        if v.notna().all() and t_num.notna().all() and (v == t_num).all():
            excluded_features.append(c)
            print(f"Excluding {c}: exact match with {target}")

for target in targets:
    if target not in train.columns or not pd.api.types.is_numeric_dtype(train[target]):
        continue
    t_num = train[target].astype(float)
    if t_num.isna().any():
        continue
    cand = [c for c in feature_cols if c not in excluded_features
            and pd.api.types.is_numeric_dtype(train[c])
            and train[c].notna().all()
            and (train[c].astype(float) >= 0).all()
            and (train[c].astype(float) <= t_num).all()]
    if not cand:
        continue
    arr = train[cand].astype(float).values
    t_arr = t_num.values
    pos_max = int(t_arr.argmax())
    max_at = float(arr[pos_max].max()) if arr.shape[1] > 0 else 0.0
    t_max_val = float(t_arr[pos_max])
    if len(cand) <= 200 and 2 * max_at >= t_max_val:
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                if (arr[:, i] + arr[:, j] == t_arr).all():
                    for c in (cand[i], cand[j]):
                        if c not in excluded_features:
                            excluded_features.append(c)
                            print(f"Excluding {c}: sum equals {target}")
    if len(cand) <= 30 and 3 * max_at >= t_max_val:
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                for k in range(j + 1, len(cand)):
                    if (arr[:, i] + arr[:, j] + arr[:, k] == t_arr).all():
                        for c in (cand[i], cand[j], cand[k]):
                            if c not in excluded_features:
                                excluded_features.append(c)

feature_cols = [c for c in feature_cols if c not in excluded_features]

for target in targets:
    if target not in train.columns or not pd.api.types.is_numeric_dtype(train[target]):
        continue
    t_num = train[target].astype(float)
    hi = []
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(train[c]):
            continue
        corr = train[c].astype(float).corr(t_num)
        if pd.notna(corr) and abs(corr) >= 0.98:
            hi.append(c)
    if hi:
        print(f"Warning: {len(hi)} features with |corr|>=0.98 vs {target}")

numeric_cols, categorical_cols, drop_cols = [], [], []
n = len(train)
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt_tr = pd.to_datetime(s, errors='coerce')
        dt_va = pd.to_datetime(val[c], errors='coerce') if c in val.columns else pd.Series(pd.NaT, index=val.index)
        for part in ['year', 'month', 'dayofweek']:
            name = f'{c}_{part}'
            train[name] = getattr(dt_tr.dt, part)
            val[name] = getattr(dt_va.dt, part)
            numeric_cols.append(name)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique() > 0.05 * n:
            drop_cols.append(c)
        else:
            categorical_cols.append(c)
if drop_cols:
    print(f"Dropping {len(drop_cols)} high-cardinality text columns")

transformers = []
if numeric_cols:
    transformers.append(('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_cols))
if categorical_cols:
    transformers.append(('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), categorical_cols))

preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
approx_width = len(numeric_cols) + sum(int(train[c].nunique(dropna=True)) for c in categorical_cols)
if approx_width <= 300:
    preprocessor.set_output(transform='pandas')

X_train = train[numeric_cols + categorical_cols]
X_val = val[numeric_cols + categorical_cols]

mlca_runtime.fit_record(preprocessor, X_train, 'train')
X_val_trans = preprocessor.transform(X_val)
print(f"Train: {X_train.shape}, Val transformed: {X_val_trans.shape}")

mlca_runtime.save_model(preprocessor, 'preprocessor.pkl')

import mlca_runtime
import pandas as pd
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

available = list(protocol.keys())
for key in ['targets', 'excluded_columns', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing {key}; available: {available}")
if 'group_column' not in protocol:
    raise KeyError(f"Missing group_column; available: {available}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]

key_like = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=True) == len(s):
            key_like.append(c)
    elif s.dtype.kind in 'iu':
        if s.nunique(dropna=True) == len(s):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

deriv_exclude = set()
deriv_reasons = {}
for target in targets:
    y = train[target]
    for c in feature_cols:
        if (train[c] == y).all():
            deriv_exclude.add(c)
            deriv_reasons[c] = f"exact equality with {target}"
    nonneg = [c for c in feature_cols if (train[c].dropna() >= 0).all()]
    for i in range(len(nonneg)):
        for j in range(i + 1, len(nonneg)):
            cols2 = [nonneg[i], nonneg[j]]
            if (train[cols2].sum(axis=1) == y).all():
                for c in cols2:
                    deriv_exclude.add(c)
                    deriv_reasons[c] = f"sum with {cols2} equals {target}"
            for k in range(j + 1, len(nonneg)):
                cols3 = [nonneg[i], nonneg[j], nonneg[k]]
                if (train[cols3].sum(axis=1) == y).all():
                    for c in cols3:
                        deriv_exclude.add(c)
                        deriv_reasons[c] = f"sum with {cols3} equals {target}"
feature_cols = [c for c in feature_cols if c not in deriv_exclude]

warnings = []
if len(targets) == 1 and pd.api.types.is_numeric_dtype(train[targets[0]]):
    y = train[targets[0]]
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            corr = train[c].corr(y)
            if abs(corr) >= 0.98:
                warnings.append(c)

ph_cols = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        for v in s.dropna().unique():
            if isinstance(v, str):
                stripped = v.strip()
                if stripped in {'?', '-', 'NA', 'N/A', ''} or v != stripped:
                    ph_cols.append(c)
                    break

num, cat, drop = [], [], []
n = len(train)
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        uniq_vals = set(s.dropna().unique())
        if uniq_vals.issubset({0, 1}):
            num.append(c)
        elif s.nunique(dropna=True) <= 0.05 * n:
            cat.append(c)
        else:
            num.append(c)
    else:
        if s.nunique(dropna=True) <= 0.05 * n:
            cat.append(c)
        else:
            drop.append(c)

transformers = []
if num:
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    num_pipe.set_output(transform='pandas')
    transformers.append(('num', num_pipe, num))
if cat:
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    cat_pipe.set_output(transform='pandas')
    transformers.append(('cat', cat_pipe, cat))

preprocessor = ColumnTransformer(transformers, remainder='drop')
preprocessor.set_output(transform='pandas')

X_train = train[feature_cols]
y_train = train[targets[0]] if targets else None
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, 'train', y=y_train)
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

print(f"Features: {len(feature_cols)}; numeric: {len(num)}; categorical: {len(cat)}; dropped: {len(drop)}")
print(f"Key-like excluded: {key_like if key_like else 'none'}")
print(f"Excluded target derivations: {deriv_reasons if deriv_reasons else 'none'}")
print(f"High corr (>=0.98): {warnings if warnings else 'none'}")
print(f"Placeholders: {len(set(ph_cols))} columns {list(set(ph_cols)) if ph_cols else 'none'}")

import json, itertools
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_integer_dtype

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = set(protocol['excluded_columns'])
group_col = protocol.get('group_column')
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
target = targets[0]
y = train[target]

# Candidate feature columns: exclude targets, protocol exclusions, group, and key-like columns.
cands = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = train[c]
    if (s.dtype == object or pd.api.types.is_string_dtype(s)) and s.notna().all() and s.nunique() == len(train):
        continue
    if is_integer_dtype(s) and s.notna().all() and s.nunique() == len(train):
        if (s.is_monotonic_increasing or s.is_monotonic_decreasing) or any(x in c.lower() for x in ('id', 'code', 'index', 'key')):
            continue
    cands.append(c)

# Drop datetime columns and high-cardinality text/code columns (one-hot is unsuitable).
feature_cols = []
for c in cands:
    s = train[c]
    if is_datetime64_any_dtype(s):
        continue
    if s.dtype == object or pd.api.types.is_string_dtype(s) or str(s.dtype) == 'category':
        if s.nunique(dropna=True) > 0.05 * len(train):
            continue
    feature_cols.append(c)

# Placeholder scan for text columns.
PLACEHOLDERS = {'?', '-', 'NA', 'N/A', ''}
ph_cols, ph_found = 0, set()
for c in feature_cols:
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s) or str(s.dtype) == 'category':
        st = s.dropna().astype(str).str.strip()
        hit = set(st[st.isin(PLACEHOLDERS)].unique())
        if hit:
            ph_cols += 1
            ph_found |= hit
if ph_cols:
    print(f"placeholders: {sorted(ph_found)} in {ph_cols} columns")

# Derivation screen: exact equality with target, or target equals sum of up to 3 non-negative features.
excl_deriv = {}
for c in feature_cols:
    if (train[c].astype(object) == y.astype(object)).all():
        excl_deriv[c] = 'equals target'
y_num = pd.to_numeric(y, errors='coerce')
y_numeric = y_num.notna().all()
num_cols = [c for c in feature_cols if is_numeric_dtype(train[c]) and c not in excl_deriv]
if y_numeric and num_cols:
    yv = y_num.astype(float)
    nonneg = [c for c in num_cols if train[c].min() >= 0 and train[c].max() <= yv.max()]
    max_k = 3 if len(nonneg) <= 20 else 2 if len(nonneg) <= 40 else 0
    if max_k:
        Xn = train[nonneg].astype(float)
        for k in range(2, max_k + 1):
            for combo in itertools.combinations(nonneg, k):
                if (Xn[list(combo)].sum(axis=1) - yv).abs().max() <= 1e-9:
                    for c in combo:
                        excl_deriv[c] = f'component of sum {combo} equals target'
                    break
            if excl_deriv:
                pass
feature_cols = [c for c in feature_cols if c not in excl_deriv]
if excl_deriv:
    print("excluded derivations: " + ", ".join(f"{a} ({b})" for a, b in list(excl_deriv.items())[:3]))

# Correlation warning for numeric outcomes only; keep flagged features.
num_cols = [c for c in feature_cols if is_numeric_dtype(train[c])]
if y_numeric and num_cols:
    corrs = train[num_cols].corrwith(y_num.astype(float)).abs()
    warn = corrs[corrs >= 0.98].index.tolist()
    if warn:
        print("high corr with target (kept): " + ", ".join(warn[:5]))

# Build preprocessing.
num_features = [c for c in feature_cols if is_numeric_dtype(train[c])]
cat_features = [c for c in feature_cols if c not in num_features]
MISSING = '__MLCA_MISSING__'
width = len(num_features) + sum(train[c].nunique(dropna=True) + 1 for c in cat_features)
dense = width <= 300
sparse_output = not dense

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
    ('scaler', StandardScaler()),
])
steps = [
    ('nan', SimpleImputer(strategy='constant', fill_value=MISSING, missing_values=np.nan, keep_empty_features=True)),
    ('strip', FunctionTransformer(np.char.strip, feature_names_out='one-to-one')),
]
for i, tok in enumerate(sorted(PLACEHOLDERS)):
    steps.append((f'ph{i}', SimpleImputer(missing_values=tok, strategy='constant', fill_value=MISSING, keep_empty_features=True)))
steps.append(('impute', SimpleImputer(missing_values=MISSING, strategy='most_frequent', keep_empty_features=True)))
steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=sparse_output)))
cat_pipe = Pipeline(steps)

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_features),
    ('cat', cat_pipe, cat_features),
], remainder='drop')
if dense:
    preprocessor.set_output(transform='pandas')
    num_pipe.set_output(transform='pandas')
    cat_pipe.set_output(transform='pandas')

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, 'train')
Xt = preprocessor.transform(X_val)
print(f"features {len(feature_cols)} (num {len(num_features)}, cat {len(cat_features)}); val transformed {Xt.shape}")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

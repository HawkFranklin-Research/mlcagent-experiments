import mlca_runtime, json, pandas as pd, numpy as np, operator
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp['train'])
proto = json.loads(inp['_protocol'].read_text())
targets = proto['targets']
excluded = proto['excluded_columns']
group = proto.get('group_column', None)
target = targets[0]

feat = [c for c in train.columns if c not in targets and c not in excluded and c != group]
drop_key = [c for c in feat if (pd.api.types.is_integer_dtype(train[c].dtype) or pd.api.types.is_object_dtype(train[c].dtype) or pd.api.types.is_string_dtype(train[c].dtype)) and train[c].nunique() == len(train)]
for c in drop_key: feat.remove(c)
print(f"Kept {len(feat)} raw features, dropped {len(drop_key)} key-like")

y = train[target]
deriv = set()
for c in feat:
    if (train[c] == y).all(): deriv.add(c)
num_nonneg = [c for c in feat if c not in deriv and pd.api.types.is_numeric_dtype(train[c]) and (train[c] >= 0).all()]
for r in [2,3]:
    for combo in combinations(num_nonneg, r):
        if any(c in deriv for c in combo): continue
        if (train[list(combo)].sum(axis=1) == y).all(): deriv.update(combo)
if deriv: print("Excluded derivations: " + ", ".join(deriv))
feat = [c for c in feat if c not in deriv]

if pd.api.types.is_numeric_dtype(y):
    warn = [c for c in feat if pd.api.types.is_numeric_dtype(train[c]) and abs(train[c].corr(y)) >= 0.98]
    if warn: print("High corr with target (kept): " + ", ".join(warn))

num_cols, cat_cols, drop_cols = [], [], []
for c in feat:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s.dtype):
        num_cols.append(c)
    elif pd.api.types.is_datetime64_any_dtype(s.dtype):
        drop_cols.append(c)
    else:
        if s.nunique() > 0.05 * len(train): drop_cols.append(c)
        else: cat_cols.append(c)
feat = [c for c in feat if c not in drop_cols]

ph_cols = [c for c in cat_cols if train[c].astype(str).str.strip().isin(['?','-','NA','N/A']).any() or train[c].astype(str).str.strip().eq('').any()]
print(f"Placeholders in {len(ph_cols)} columns" + (": " + ", ".join(ph_cols) if ph_cols else ""))

X = train[feat]
num_cols = [c for c in num_cols if c in X.columns]
cat_cols = [c for c in cat_cols if c in X.columns]
n_num = len(num_cols)
n_cat = sum(X[c].nunique() for c in cat_cols) if cat_cols else 0
use_sparse = (n_num + n_cat) > 300

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipe = Pipeline([
    ('strip', FunctionTransformer(operator.methodcaller('replace', r'^\s+|\s+$', '', regex=True))),
    ('placeholders', FunctionTransformer(operator.methodcaller('replace', ['?','-','NA','N/A',''], np.nan))),
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=use_sparse)),
])
transformers = []
if num_cols: transformers.append(('num', num_pipe, num_cols))
if cat_cols: transformers.append(('cat', cat_pipe, cat_cols))
preprocessor = ColumnTransformer(transformers, remainder='drop')
if not use_sparse:
    preprocessor.set_output(transform='pandas')
    num_pipe.set_output(transform='pandas')
    cat_pipe.set_output(transform='pandas')

mlca_runtime.fit_record(preprocessor, X, partition='train')
mlca_runtime.save_model(preprocessor, 'preprocessor.pkl')
print(f"Preprocessor saved: {n_num} numeric, {len(cat_cols)} categorical, width={n_num+n_cat}")

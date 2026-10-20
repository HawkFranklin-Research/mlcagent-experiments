import mlca_runtime
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
for key in ['targets', 'excluded_columns', 'group_column', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing {key}; available: {list(protocol.keys())}")
targets = list(protocol['targets'])
excluded = list(protocol['excluded_columns'])
group_col = protocol['group_column']
train = pd.read_parquet(inputs['train'])
missing = [t for t in targets if t not in train.columns]
if missing:
    raise KeyError(f"Missing target columns: {missing}")

feat_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if s.dtype == object or str(s.dtype) == 'string' or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) == len(train):
            continue
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) == len(train):
        if any(tok in c.lower() for tok in ('id', 'key', 'code')):
            continue
    feat_cols.append(c)
if not feat_cols:
    raise ValueError("No feature columns remain")

max_card = max(1, int(0.05 * len(train)))
numeric_cols, cat_cols = [], []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_bool_dtype(s.dtype):
        numeric_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s.dtype):
        numeric_cols.append(c)
    elif s.dtype == object or str(s.dtype) == 'string' or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) <= max_card:
            cat_cols.append(c)

total_width = len(numeric_cols) + sum(train[c].nunique(dropna=False) for c in cat_cols)
sparse_output = total_width > 300
numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median'))])
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=sparse_output))
])
transformers = []
if numeric_cols:
    transformers.append(('num', numeric_transformer, numeric_cols))
if cat_cols:
    transformers.append(('cat', cat_transformer, cat_cols))
preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder='drop',
    sparse_threshold=1.0 if sparse_output else 0.0
)
if not sparse_output:
    preprocessor.set_output(transform='pandas')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=100, random_state=42)))
])
fitted = mlca_runtime.fit_record(pipeline, train[feat_cols], 'train', y=train[targets])
mlca_runtime.save_model(fitted, "alternative_model.pkl")
print("Fitted alternative model and saved alternative_model.pkl")

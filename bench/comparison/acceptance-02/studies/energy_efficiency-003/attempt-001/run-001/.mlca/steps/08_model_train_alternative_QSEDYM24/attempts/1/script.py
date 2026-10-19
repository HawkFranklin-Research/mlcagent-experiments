import mlca_runtime
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ['targets', 'excluded_columns', 'group_column', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

train = pd.read_parquet(inputs["train"])
print(f"Train shape: {train.shape}")

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

drop_key = []
for c in feature_cols:
    s = train[c]
    if s.dtype == 'object' or s.dtype.name == 'string':
        if s.nunique() == len(s):
            drop_key.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique() == len(s) and s.min() == 0 and s.max() == len(s) - 1:
            drop_key.append(c)
if drop_key:
    print(f"Dropping key-like columns: {drop_key}")
    feature_cols = [c for c in feature_cols if c not in drop_key]

numeric_cols, categorical_cols = [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s):
        if s.nunique() <= max(10, int(0.05 * len(s))):
            categorical_cols.append(c)
        else:
            print(f"Dropping high-cardinality column: {c} (nunique={s.nunique()})")
    else:
        try:
            pd.to_numeric(s)
            numeric_cols.append(c)
        except Exception:
            print(f"Dropping unhandled column: {c} dtype={s.dtype}")

assert not any(t in numeric_cols + categorical_cols for t in targets)
assert not any(e in numeric_cols + categorical_cols for e in excluded)
print(f"Numeric features: {len(numeric_cols)}, Categorical features: {len(categorical_cols)}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop'
)
preprocessor.set_output(transform='pandas')

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train = train[feature_cols]
y_train = train[targets]

mlca_runtime.fit_record(pipeline, X_train, partition='train', y=y_train)
mlca_runtime.save_model(pipeline, "alternative_model.pkl")
print("Saved alternative_model.pkl")

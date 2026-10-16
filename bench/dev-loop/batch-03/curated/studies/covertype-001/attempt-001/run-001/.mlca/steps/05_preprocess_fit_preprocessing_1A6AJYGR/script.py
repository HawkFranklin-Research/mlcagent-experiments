import json
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
task = protocol["task"]

all_cols = list(train.columns)

def is_key_like(series, name):
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return series.nunique(dropna=True) == len(series)
    if pd.api.types.is_integer_dtype(series):
        if series.nunique(dropna=True) == len(series):
            if 'id' in name.lower() or 'code' in name.lower():
                return True
            if series.max() - series.min() + 1 == len(series):
                return True
    return False

feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group and not is_key_like(train[c], c)]

target = targets[0]
excluded_derivations = []
for feat in feature_cols:
    if (train[feat] == train[target]).all():
        excluded_derivations.append((feat, "exact equality with target"))
feature_cols = [c for c in feature_cols if c not in [x[0] for x in excluded_derivations]]

text_cols = [c for c in feature_cols if train[c].dtype == object or pd.api.types.is_string_dtype(train[c])]
placeholders = ['?', '-', 'NA', 'N/A']
found = {}
for c in text_cols:
    vals = train[c].astype(str).str.strip()
    for p in placeholders:
        if (vals == p).any():
            found.setdefault(p, []).append(c)
    if (vals == '').any():
        found.setdefault('whitespace-only', []).append(c)
placeholder_info = f"{found if found else 'none'}; {len(found)} columns"

numeric_cols = []
categorical_cols = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        numeric_cols.append(c)
    else:
        if train[c].nunique(dropna=True) <= 0.05 * len(train):
            categorical_cols.append(c)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
numeric_transformer.set_output(transform='pandas')

transformers = [('num', numeric_transformer, numeric_cols)]
if categorical_cols:
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    categorical_transformer.set_output(transform='pandas')
    transformers.append(('cat', categorical_transformer, categorical_cols))

preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
preprocessor.set_output(transform='pandas')

X_train = train[feature_cols]
y_train = train[target]
X_val = val[feature_cols]

preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train", y_train)
X_val_trans = preprocessor.transform(X_val)

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", len(feature_cols), "train")
mlca_runtime.record_metric("n_encoded_columns", X_val_trans.shape[1], "val")

print(f"Features: {len(feature_cols)}")
print(f"Excluded derivations: {excluded_derivations}")
print(f"Placeholders: {placeholder_info}")
print(f"Train shape: {X_train.shape}, Val transformed shape: {X_val_trans.shape}")
print("Saved preprocessor.pkl")

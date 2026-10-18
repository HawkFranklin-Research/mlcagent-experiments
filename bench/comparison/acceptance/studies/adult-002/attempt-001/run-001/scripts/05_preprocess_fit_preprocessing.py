import json
import numpy as np
import pandas as pd
import mlca_runtime
from functools import partial
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)
if not targets:
    raise ValueError(f"No targets in protocol; keys: {list(protocol.keys())}")
target = targets[0]

all_cols = list(train.columns)

def is_key_like(s):
    if s.dtype == 'object':
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=True) == len(s):
            vals = s.dropna().sort_values().values
            if len(vals) == len(s) and np.array_equal(vals, np.arange(vals.min(), vals.max()+1)):
                return True
    return False

key_cols = [c for c in all_cols if is_key_like(train[c])]
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and (group is None or c != group) and c not in key_cols]

deriv_cols = []
for c in feature_cols:
    if train[c].equals(train[target]):
        deriv_cols.append(c)
        print(f"Excluded feature {c}: exact copy of target")
feature_cols = [c for c in feature_cols if c not in deriv_cols]

placeholder_tokens = {'?', '-', 'NA', 'N/A', ''}
placeholder_cols = set()
for c in feature_cols:
    if train[c].dtype == 'object':
        for v in train[c].dropna().unique():
            if isinstance(v, str) and v.strip() in placeholder_tokens:
                placeholder_cols.add(c)
                break
if placeholder_cols:
    print(f"Placeholders found in {len(placeholder_cols)} columns: {sorted(placeholder_cols)}")
else:
    print("No placeholders found")

numeric_cols = []
cat_cols = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c]):
        numeric_cols.append(c)
    else:
        cat_cols.append(c)

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_steps = []
if placeholder_cols:
    cat_steps.append(('strip', FunctionTransformer(partial(pd.DataFrame.replace, to_replace=r'^\s+|\s+$', value='', regex=True))))
    cat_steps.append(('placeholder', FunctionTransformer(partial(pd.DataFrame.replace, to_replace=r'^(\?|-|NA|N/A|)$', value=np.nan, regex=True))))
cat_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
cat_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
cat_transformer = Pipeline(cat_steps)

preprocessor = ColumnTransformer([
    ('num', num_transformer, numeric_cols),
    ('cat', cat_transformer, cat_cols)
])
preprocessor.set_output(transform='pandas')

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()

mlca_runtime.fit_record(preprocessor, X_train, "train", y=None)
X_val_trans = preprocessor.transform(X_val)
print(f"Transformed val shape: {X_val_trans.shape}")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

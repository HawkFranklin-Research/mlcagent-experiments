import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in proto:
        raise KeyError(f"Missing {k}; available: {list(proto.keys())}")
targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto["group_column"]
task = proto["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop_cols = set(targets) | set(excluded) | ({group} if group else set())
feat_cols = [c for c in train.columns if c not in drop_cols]

key_like = []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=False) == len(s):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(s) and s.sort_values().diff().dropna().eq(1).all():
            key_like.append(c)
feat_cols = [c for c in feat_cols if c not in key_like]

y_train = train[targets[0]]
y_num = pd.Series(pd.factorize(y_train)[0], index=train.index) if not pd.api.types.is_numeric_dtype(y_train) else y_train
derived = []
for c in feat_cols:
    if train[c].equals(y_train) or (pd.api.types.is_numeric_dtype(train[c]) and train[c].equals(y_num)):
        derived.append(c)
num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(train[c]) and (train[c] >= 0).all()]
for i in range(len(num_cols)):
    for j in range(i + 1, len(num_cols)):
        if (train[num_cols[i]] + train[num_cols[j]]).equals(y_num):
            derived.append(num_cols[i] + "+" + num_cols[j])
        for k in range(j + 1, len(num_cols)):
            if (train[num_cols[i]] + train[num_cols[j]] + train[num_cols[k]]).equals(y_num):
                derived.append(num_cols[i] + "+" + num_cols[j] + "+" + num_cols[k])
derived = list(set(derived))
feat_cols = [c for c in feat_cols if c not in derived]

warn = []
for c in feat_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        corr = train[c].corr(y_num)
        if abs(corr) >= 0.98:
            warn.append(c)
if warn:
    print(f"High corr >=0.98: {warn}")

placeholders = ['?', '-', 'NA', 'N/A']
ph_cols = []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype(str).str.strip()
        if stripped.isin(placeholders + ['']).any() or stripped.str.match(r'^\s*$').any():
            ph_cols.append(c)
if ph_cols:
    print(f"Placeholders found in {len(ph_cols)} columns: {ph_cols}")

numeric_cols = []
cat_cols = []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=False) > 0.05 * len(train):
            continue
        cat_cols.append(c)

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, placeholders=('?', '-', 'NA', 'N/A')):
        self.placeholders = placeholders
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for c in X.columns:
            if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_string_dtype(X[c]):
                X[c] = X[c].astype(str).str.strip()
                X[c] = X[c].replace(list(self.placeholders) + [''], np.nan)
                X[c] = X[c].replace(r'^\s*$', np.nan, regex=True)
        return X
    def get_feature_names_out(self, input_features=None):
        return np.array(input_features if input_features is not None else self.feature_names_in_)

expected_width = len(numeric_cols) + sum(train[c].nunique(dropna=False) for c in cat_cols)
sparse = expected_width > 300
numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_pipe = Pipeline([
    ('cleaner', TextCleaner()),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=sparse)),
])
preprocessor = ColumnTransformer(
    [('num', numeric_pipe, numeric_cols), ('cat', categorical_pipe, cat_cols)],
    remainder='drop',
)
if not sparse:
    preprocessor.set_output(transform='pandas')
    numeric_pipe.set_output(transform='pandas')
    categorical_pipe.set_output(transform='pandas')

X_train = train[feat_cols]
X_val = val[feat_cols]
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train", y=y_train)
X_val_trans = preprocessor.transform(X_val)
print(f"Train {X_train.shape}, Val {X_val.shape}, transformed {X_val_trans.shape}")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

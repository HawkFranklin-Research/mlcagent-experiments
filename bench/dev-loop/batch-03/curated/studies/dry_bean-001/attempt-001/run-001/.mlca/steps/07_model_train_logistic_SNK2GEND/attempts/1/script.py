import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(proto, key):
    if key not in proto:
        raise KeyError(f"Missing {key}. Available top-level keys: {list(proto.keys())}")
    return proto[key]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group_col = need(protocol, "group_column")
task = need(protocol, "task")

train = pd.read_parquet(inputs["train"])
drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)

feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if pd.api.types.is_float_dtype(s.dtype):
        feature_cols.append(c)
    elif pd.api.types.is_integer_dtype(s.dtype):
        if s.nunique(dropna=False) == len(s):
            continue
        feature_cols.append(c)
    elif pd.api.types.is_object_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) == len(s):
            continue
        feature_cols.append(c)
    else:
        feature_cols.append(c)

X_train = train[feature_cols].copy()
y_train = train[targets[0]] if len(targets) == 1 else train[targets]

with open(inputs["05_preprocess_fit_preprocessing_Y4SGGFB6/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_select_features_W4621TGT/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

def input_names(obj):
    return getattr(obj, "feature_names_in_", None)

pre_names = input_names(preprocessor)
sel_names = input_names(selector)

use_preprocessor = True
if sel_names is not None:
    if set(sel_names) == set(feature_cols):
        use_preprocessor = False
else:
    if isinstance(selector, Pipeline):
        first_step = list(selector.named_steps.values())[0]
        if isinstance(first_step, ColumnTransformer):
            use_preprocessor = False
        elif isinstance(first_step, Pipeline) and any(isinstance(s, ColumnTransformer) for s in first_step.named_steps.values()):
            use_preprocessor = False

if use_preprocessor:
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            n_out = len(preprocessor.get_feature_names_out())
            if n_out <= 300 and hasattr(preprocessor, "set_output"):
                preprocessor.set_output(transform="pandas")
        except Exception:
            pass
    if pre_names is not None:
        X_train = X_train[list(pre_names)]
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("selector", selector),
        ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
    ])
else:
    if sel_names is not None:
        X_train = X_train[list(sel_names)]
    pipeline = Pipeline([
        ("selector", selector),
        ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
    ])

pipeline = mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)
mlca_runtime.save_model(pipeline, "model.pkl")
print(f"features: {len(feature_cols)}")
print(f"train rows: {len(X_train)}")

import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

with open(inputs["_protocol"], "r") as f:
    protocol = json.load(f)

def req(proto, key):
    if key not in proto:
        raise KeyError(f"Missing protocol key '{key}'. Available keys: {list(proto.keys())}")
    return proto[key]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
group_col = req(protocol, "group_column")
task = req(protocol, "task")

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_BKKYG4P6/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_select_features_GQKA7M4V/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

def get_input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def is_key_like(series):
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        return series.nunique(dropna=False) == len(series) and series.nunique(dropna=False) > 1
    if pd.api.types.is_integer_dtype(series):
        return series.nunique(dropna=False) == len(series) and series.nunique(dropna=False) > 1
    return False

exclude = set(targets) | set(excluded)
if group_col:
    exclude.add(group_col)
key_names = {"id", "unnamed: 0", "index"}
feature_cols = [
    c for c in train.columns
    if c not in exclude
    and c.lower() not in key_names
    and not is_key_like(train[c])
]

def selector_includes_preprocessing(sel, raw_cols):
    sel_in = get_input_cols(sel)
    if sel_in is not None and set(sel_in) == set(raw_cols):
        return True
    if isinstance(sel, Pipeline):
        steps = list(sel.named_steps.items())
        if steps:
            first_name, first_step = steps[0]
            if isinstance(first_step, pd.DataFrame) or first_step.__class__.__name__ == "ColumnTransformer":
                return True
            if hasattr(first_step, "feature_names_in_") and set(first_step.feature_names_in_) == set(raw_cols):
                return True
    return False

raw_feature_cols = feature_cols[:]
sel_in = get_input_cols(selector)
pre_in = get_input_cols(preprocessor)

if selector_includes_preprocessing(selector, raw_feature_cols):
    steps = [("selector", selector)]
    if sel_in:
        raw_feature_cols = list(sel_in)
else:
    steps = [("preprocessor", preprocessor), ("selector", selector)]
    if pre_in:
        raw_feature_cols = list(pre_in)

steps.append(("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)))
pipeline = Pipeline(steps)

try:
    front = Pipeline(steps[:-1])
    transformed = front.transform(train[raw_feature_cols].head(5))
    n_cols = transformed.shape[1] if hasattr(transformed, "shape") else len(transformed[0])
    if n_cols <= 300:
        pipeline.set_output(transform="pandas")
except Exception:
    pass

X_train = train[raw_feature_cols].copy()
y_train = train[targets[0]] if len(targets) == 1 else train[targets]

fitted = mlca_runtime.fit_record(pipeline, X_train, partition="train", y=y_train)
mlca_runtime.save_model(fitted, "model.pkl")

print(f"train rows: {train.shape[0]}")
print(f"feature columns: {len(raw_feature_cols)}")
print("saved: model.pkl")

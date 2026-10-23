import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column", None)
train = pd.read_parquet(inputs["train"])

def is_key_like(s):
    n = len(s)
    if n == 0:
        return False
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(s.dtype) and s.is_unique:
        mn, mx = s.min(), s.max()
        if (mn == 0 or mn == 1) and mx == mn + n - 1:
            return True
    return False

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    if is_key_like(train[c]):
        continue
    feature_cols.append(c)

with open(inputs["05_preprocess_fit_preprocessing_6FVQ17HJ/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_select_features_R6V6W10Y/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

selector_wraps = isinstance(selector, Pipeline) and len(selector.named_steps) > 1
frontend = selector if (selector is not None and selector_wraps) else preprocessor

if frontend is not None and hasattr(frontend, "feature_names_in_"):
    expected = list(frontend.feature_names_in_)
    missing = [c for c in expected if c not in feature_cols]
    if missing:
        raise ValueError(f"Saved frontend expects columns outside allowed feature set: {missing}")
    input_cols = expected
else:
    input_cols = feature_cols

X_train = train[input_cols]
y_train = train[targets[0]]

steps = []
if selector is not None and selector_wraps:
    steps.append(("selector", selector))
else:
    if preprocessor is not None:
        steps.append(("preprocessor", preprocessor))
    if selector is not None:
        steps.append(("selector", selector))
steps.append(("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)))

pipeline = Pipeline(steps)
fitted = mlca_runtime.fit_record(pipeline, X_train, "train", y_train)
mlca_runtime.save_model(fitted, "model.pkl")
print(f"train rows: {len(train)}, features used: {len(input_cols)}")
print(f"pipeline steps: {list(fitted.named_steps)}")
print("saved model.pkl")

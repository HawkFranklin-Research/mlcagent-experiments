import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available keys: {list(d.keys())}")
    return d[k]

targets = list(req(protocol, "targets"))
excluded = list(req(protocol, "excluded_columns"))
group = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])
y = train[target]

def load_pkl(path):
    obj = pickle.loads(path.read_bytes())
    if isinstance(obj, dict):
        for k in ("preprocessor", "selector", "pipeline", "model"):
            if k in obj:
                return obj[k]
    return obj

pre = load_pkl(inputs["05_preprocess_fit_preprocessing_6Y57Q816/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_select_features_HZT7Z0Q4/selector.pkl"])

raw_features = [c for c in train.columns if c not in targets and c not in excluded and c != group]
feature_cols = []
n = len(train)
for c in raw_features:
    s = train[c]
    if (s.dtype == object or pd.api.types.is_string_dtype(s)):
        if s.notna().sum() == n and s.nunique(dropna=True) == n:
            continue
    feature_cols.append(c)

pre_inputs = getattr(pre, "feature_names_in_", None)
if pre_inputs is not None:
    pre_inputs = list(pre_inputs)
    for c in pre_inputs:
        if c not in train.columns:
            raise ValueError(f"Preprocessor expects missing column: {c}")
        if c in targets or c in excluded or c == group:
            raise ValueError(f"Preprocessor input includes forbidden column: {c}")
    feature_cols = pre_inputs

sel_inputs = getattr(sel, "feature_names_in_", None)
sel_includes_pre = False
if sel_inputs is not None and pre_inputs is not None:
    sel_includes_pre = list(sel_inputs) == list(pre_inputs)
if isinstance(sel, Pipeline):
    for step in sel.named_steps.values():
        if step.__class__.__name__ == "ColumnTransformer":
            sel_includes_pre = True
        step_inputs = getattr(step, "feature_names_in_", None)
        if step_inputs is not None and pre_inputs is not None:
            if list(step_inputs) == list(pre_inputs):
                sel_includes_pre = True

model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
if sel_includes_pre:
    pipeline = Pipeline([("selector", sel), ("classifier", model)])
else:
    pipeline = Pipeline([("preprocessor", pre), ("selector", sel), ("classifier", model)])

X = train.loc[:, feature_cols]
mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(pipeline, "model.pkl")

print(f"train shape: {train.shape}")
print(f"features: {len(feature_cols)}")
print(f"selector includes preprocessing: {sel_includes_pre}")
print("saved: model.pkl")

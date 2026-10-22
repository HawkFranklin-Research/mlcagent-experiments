import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
task = protocol["task"]

train = pd.read_parquet(inputs["train"])

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_6GF5BX9F/preprocessor.pkl")
sel = load_pkl("06_features_select_features_1B1C3SBH/selector.pkl")

def input_names(obj):
    names = getattr(obj, "feature_names_in_", None)
    if names is not None:
        return list(names)
    if isinstance(obj, Pipeline):
        for _, step in obj.named_steps.items():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                return list(names)
    return None

input_cols = input_names(pre)
if input_cols is None:
    input_cols = input_names(sel)
if input_cols is None:
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    input_cols = [c for c in train.columns if c not in drop]

bad = set(input_cols) & (set(targets) | set(excluded))
if bad:
    raise ValueError(f"Feature columns include target/excluded: {sorted(bad)}")
missing = set(input_cols) - set(train.columns)
if missing:
    raise KeyError(f"Missing expected feature columns: {sorted(missing)}")

X = train[input_cols]
y = train[targets[0]]

def preprocessor_like(step):
    name = type(step).__name__
    return name in ("ColumnTransformer", "FeatureUnion") or isinstance(step, Pipeline) or hasattr(step, "transformers_")

sel_includes_pre = isinstance(sel, Pipeline) and any(
    preprocessor_like(step) for _, step in sel.named_steps.items()
)

model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
if sel_includes_pre:
    pipe = Pipeline([("frontend", sel), ("model", model)])
else:
    pipe = Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])

mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(pipe, "model.pkl")
print("train_rows", len(X), "features", len(input_cols), "selector_wraps_pre", sel_includes_pre, "saved model.pkl")

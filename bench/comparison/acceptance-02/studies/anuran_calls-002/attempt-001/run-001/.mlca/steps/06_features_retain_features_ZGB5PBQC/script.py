import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())


def get_key(d, key):
    if key not in d:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {sorted(d.keys())}")
    return d[key]


targets = list(get_key(proto, "targets"))
excluded = list(get_key(proto, "excluded_columns"))
group_col = get_key(proto, "group_column")
_ = get_key(proto, "task")

train = pd.read_parquet(inputs["train"])
for c in targets + [group_col]:
    if c not in train.columns:
        raise KeyError(f"Required column {c!r} missing from training table")

base = set(targets) | set(excluded) | {group_col}


def is_keylike(s):
    if len(s) == 0 or s.nunique(dropna=False) != len(s):
        return False
    return (
        pd.api.types.is_integer_dtype(s.dtype)
        or pd.api.types.is_string_dtype(s.dtype)
        or pd.api.types.is_object_dtype(s.dtype)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )


table_features = [
    c for c in train.columns
    if c not in base and not is_keylike(train[c])
]

with open(inputs["05_preprocess_fit_preprocessing_A8X1QVXT/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, "feature_names_in_"):
    expected = list(preprocessor.feature_names_in_)
    if set(expected) != set(table_features):
        missing = sorted(set(expected) - set(table_features))
        extra = sorted(set(table_features) - set(expected))
        raise ValueError(f"Feature schema mismatch; missing={missing}, extra={extra}")
    feature_cols = expected
else:
    feature_cols = table_features

X = train[feature_cols]
selector = FunctionTransformer(validate=False)

if isinstance(preprocessor, Pipeline):
    steps = list(preprocessor.steps)
    if steps and isinstance(steps[-1][1], FunctionTransformer) and steps[-1][1].func is None:
        model = preprocessor
    else:
        existing = {name for name, _ in steps}
        name = "selector"
        i = 0
        while name in existing:
            i += 1
            name = f"selector_{i}"
        model = Pipeline(steps + [(name, selector)])
else:
    model = Pipeline([("preprocessor", preprocessor), ("selector", selector)])

if hasattr(model, "set_output"):
    try:
        Xt = preprocessor.transform(X)
        width = Xt.shape[1] if hasattr(Xt, "shape") else len(feature_cols)
        if width <= 300:
            model.set_output(transform="pandas")
    except Exception:
        pass

mlca_runtime.fit_record(model, X, "train", y=None)
mlca_runtime.save_model(model, "selector_all.pkl")
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_train_rows", float(len(X)), "train")

print(f"features: {len(feature_cols)}")
print(f"train rows: {len(X)}")
print("saved selector_all.pkl")

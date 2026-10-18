import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k!r}; available top-level keys: {sorted(d.keys())}")
    return d[k]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
task = req(protocol, "task")
group_col = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(paths["train"])
with open(paths["05_preprocess_fit_preprocessing_XT8NJ07Q/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(paths["06_features_select_features_A3NFBXN9/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

def input_cols(obj):
    return getattr(obj, "feature_names_in_", None)

def accepts_raw(obj):
    cols = input_cols(obj)
    return cols is not None and all(c in train.columns for c in cols)

def compute_feature_cols(df):
    exclude = set(targets) | set(excluded)
    if group_col:
        exclude.add(group_col)
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if s.nunique(dropna=True) == len(s):
            if pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                continue
        cols.append(c)
    return cols

if selector is not None and accepts_raw(selector):
    feature_cols = list(input_cols(selector))
    use_selector_only = True
elif preprocessor is not None:
    feature_cols = input_cols(preprocessor)
    feature_cols = list(feature_cols) if feature_cols is not None else compute_feature_cols(train)
    use_selector_only = False
elif selector is not None:
    feature_cols = compute_feature_cols(train)
    use_selector_only = False
else:
    feature_cols = compute_feature_cols(train)
    use_selector_only = False

for c in feature_cols:
    if c in targets or c in excluded or (group_col and c == group_col):
        raise ValueError(f"Feature column {c!r} is target/excluded/group")
missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise KeyError(f"Missing feature columns: {missing}")

X_train = train.loc[:, feature_cols].copy()
y_train = train[target]

model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

if use_selector_only:
    if isinstance(selector, Pipeline):
        pipeline = clone(selector)
        pipeline.steps = list(pipeline.steps) + [("model", model)]
    else:
        pipeline = Pipeline([("selector", clone(selector)), ("model", model)])
else:
    steps = []
    if preprocessor is not None:
        steps.append(("preprocessor", clone(preprocessor)))
    if selector is not None:
        steps.append(("selector", clone(selector)))
    steps.append(("model", model))
    pipeline = Pipeline(steps)
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            n_out = len(preprocessor.get_feature_names_out())
        except Exception:
            n_out = None
        if n_out is not None and n_out <= 300:
            prep = pipeline.named_steps.get("preprocessor")
            if prep is not None and hasattr(prep, "set_output"):
                prep.set_output(transform="pandas")

fitted = mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "model.pkl")
print(f"train rows: {len(train)}")
print(f"features: {len(feature_cols)}")
print("saved model.pkl")

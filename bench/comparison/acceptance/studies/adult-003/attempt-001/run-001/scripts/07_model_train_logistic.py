import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"{k} missing; available top-level keys: {list(d)}")
    return d[k]

targets = list(need(proto, "targets"))
excluded = list(need(proto, "excluded_columns"))
group = proto.get("group_column")
if len(targets) != 1:
    raise ValueError("expected one target")
target = targets[0]

train = pd.read_parquet(inp["train"])
if target not in train:
    raise KeyError(f"target {target} not in train columns")

def keylike(s):
    all_unique = s.nunique(dropna=False) == len(s)
    return all_unique and (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_integer_dtype(s)
    )

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    if keylike(train[c]):
        continue
    feature_cols.append(c)
if not feature_cols:
    raise ValueError("no feature columns")

with open(inp["05_preprocess_fit_preprocessing_599QM3CC/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_select_features_Y69A7R1X/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_names(obj):
    fn = getattr(obj, "feature_names_in_", None)
    if fn is not None:
        return list(fn)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            fn = getattr(step, "feature_names_in_", None)
            if fn is not None:
                return list(fn)
    return None

def has_step(obj, names):
    return isinstance(obj, Pipeline) and any(n in names for n, _ in obj.steps)

pre_names = input_names(pre)
sel_names = input_names(sel)
pre_has_sel = has_step(pre, {"selector", "feature_selection", "select"})
sel_has_pre = has_step(sel, {"preprocessor", "preprocess", "preprocessing", "column_transformer", "features"})

frontend = None
required = list(feature_cols)

if isinstance(sel, Pipeline) and (
    sel_has_pre or (sel_names is not None and set(sel_names) == set(feature_cols))
):
    frontend = sel
    if sel_names is not None:
        if set(sel_names) != set(feature_cols):
            raise ValueError("selector input columns differ from feature columns")
        required = list(sel_names)
elif pre_has_sel and (pre_names is None or set(pre_names) == set(feature_cols)):
    frontend = pre
    if pre_names is not None:
        required = list(pre_names)
elif pre_names is not None:
    if set(pre_names) != set(feature_cols):
        raise ValueError("preprocessor input columns differ from feature columns")
    required = list(pre_names)

if set(required) & (set(targets) | set(excluded)):
    raise ValueError("target/excluded column reached model input")

X = train[required].copy()
y = train[target]

model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
if frontend is not None:
    pipe = Pipeline([("frontend", frontend), ("model", model)])
else:
    pipe = Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])

fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)
if fitted is None:
    fitted = pipe
mlca_runtime.save_model(fitted, "model.pkl")
print(f"features={len(required)} train_rows={len(X)}")
print("saved model.pkl")

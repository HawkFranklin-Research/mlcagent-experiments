import mlca_runtime, pickle, json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []) or [])
group_col = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def load_pkl(key):
    try:
        import pickle as _p
        return _p.loads(inputs[key].read_bytes())
    except Exception:
        return None

pre = load_pkl("05_preprocess_fit_preprocessing_K97TX9BR/preprocessor.pkl")
sel = load_pkl("06_features_select_features_RVKX3HX3/selector.pkl")

def input_columns(obj):
    if obj is None:
        return None
    if isinstance(obj, Pipeline):
        try:
            return list(obj.feature_names_in_)
        except Exception:
            pass
        if obj.steps:
            try:
                return list(obj.steps[0][1].feature_names_in_)
            except Exception:
                pass
    try:
        return list(obj.feature_names_in_)
    except Exception:
        return None

def has_preprocessing(obj):
    if not isinstance(obj, Pipeline):
        return False
    for name, step in obj.steps:
        if isinstance(step, ColumnTransformer):
            return True
        if "pre" in name.lower() or "preprocess" in name.lower():
            return True
        if isinstance(step, Pipeline) and has_preprocessing(step):
            return True
    return False

def includes_pre(p, s):
    if s is None:
        return False
    if has_preprocessing(s):
        return True
    if p is None:
        return False
    pc, sc = input_columns(p), input_columns(s)
    return bool(pc and sc and set(pc) == set(sc))

if includes_pre(pre, sel):
    frontend = sel
elif pre is not None and sel is not None:
    frontend = Pipeline([("pre", pre), ("sel", sel)])
elif sel is not None:
    frontend = sel
elif pre is not None:
    frontend = pre
else:
    raise RuntimeError("No preprocessor or selector available")

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in train.columns if c not in drop]
cols = input_columns(frontend) or feature_cols
cols = [c for c in cols if c in train.columns and c not in drop]
if not cols:
    cols = feature_cols
print("features:", len(cols), "frontend:", type(frontend).__name__)

X_train, y_train = train.loc[:, cols], train[targets[0]]
X_val, y_val = val.loc[:, cols], val[targets[0]]

pipe = Pipeline([
    ("frontend", frontend),
    ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
])
pipe = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(pipe, "model.pkl")
print("fitted rows:", len(X_train), "saved model.pkl")

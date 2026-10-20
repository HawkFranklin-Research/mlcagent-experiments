import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
with open(inp["_protocol"]) as f:
    protocol = json.load(f)

try:
    targets = protocol["targets"]
    task = protocol["task"]
    excluded = protocol["excluded_columns"]
except KeyError as exc:
    raise KeyError(f"missing protocol key {exc}; available: {list(protocol)}") from exc

group = protocol.get("group_column")
train = pd.read_parquet(inp["train"])
target = targets[0]
y = train[target]

with open(inp["06_preprocess_fit_preprocessing_1DA5F11K/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["07_features_select_features_W8F8XFJ6/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_cols(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        return [str(c) for c in obj.feature_names_in_]
    if isinstance(obj, Pipeline):
        steps = list(obj.named_steps.values())
        if steps and hasattr(steps[0], "feature_names_in_"):
            return [str(c) for c in steps[0].feature_names_in_]
    return None

def same(a, b):
    return a is not None and b is not None and list(a) == list(b)

def has_preprocessor(obj):
    if not isinstance(obj, Pipeline):
        return False
    for st in obj.named_steps.values():
        if isinstance(st, ColumnTransformer):
            return True
        if isinstance(st, Pipeline) and has_preprocessor(st):
            return True
    return False

pre_in = input_cols(pre)
sel_in = input_cols(sel)
raw_cols = pre_in
if raw_cols is None and isinstance(sel, Pipeline):
    raw_cols = sel_in
if raw_cols is None:
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    raw_cols = [c for c in train.columns if c not in drop]
    for c in list(raw_cols):
        s = train[c]
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == len(s):
            raw_cols.remove(c)
        elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
            raw_cols.remove(c)

raw_cols = list(raw_cols)
missing = [c for c in raw_cols if c not in train.columns]
if missing:
    raise KeyError(f"missing feature columns: {missing}")
X = train[raw_cols].copy()

def set_pandas(obj):
    if obj is None or not hasattr(obj, "get_feature_names_out") or not hasattr(obj, "set_output"):
        return
    try:
        n = len(obj.get_feature_names_out())
    except Exception:
        return
    if n <= 300:
        try:
            obj.set_output(transform="pandas")
        except Exception:
            pass

set_pandas(pre)
if isinstance(sel, Pipeline):
    set_pandas(sel)

lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
sel_complete = isinstance(sel, Pipeline) and (
    same(sel_in, raw_cols) or has_preprocessor(sel)
)

if sel_complete:
    model = Pipeline([("selector", sel), ("model", lr)])
elif pre is not None and (same(pre_in, raw_cols) or pre_in is None):
    pre_out = None
    if hasattr(pre, "get_feature_names_out"):
        try:
            pre_out = [str(c) for c in pre.get_feature_names_out()]
        except Exception:
            pass
    if sel is None:
        model = Pipeline([("preprocessor", pre), ("model", lr)])
    elif sel_in is not None and pre_out is not None and same(sel_in, pre_out):
        model = Pipeline([("preprocessor", pre), ("selector", sel), ("model", lr)])
    elif sel_in is None and not isinstance(sel, Pipeline):
        model = Pipeline([("preprocessor", pre), ("selector", sel), ("model", lr)])
    else:
        model = Pipeline([("preprocessor", pre), ("model", lr)])
else:
    model = Pipeline([("preprocessor", pre), ("selector", sel), ("model", lr)])

mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(model, "model.pkl")
print(f"features={len(raw_cols)} steps={[n for n, _ in model.steps]}")
print("saved model.pkl")

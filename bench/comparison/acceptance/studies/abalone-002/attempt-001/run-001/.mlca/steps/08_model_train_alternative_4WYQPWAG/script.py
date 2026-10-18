import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key}; available keys: {list(protocol.keys())}")
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {list(task.keys())}")
primary_metric = task["primary_metric"]

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]

train = pd.read_parquet(inp["train"])

def is_key_like(s: pd.Series) -> bool:
    if s.dtype.kind in "OU" or str(s.dtype) == "category":
        return s.nunique(dropna=False) == len(s)
    if s.dtype.kind in "iu":
        return s.nunique(dropna=False) == len(s)
    return False

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    if is_key_like(train[c]):
        continue
    feature_cols.append(c)

with open(inp["05_preprocess_fit_preprocessing_FG7YG4JM/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_2K81ED4E/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def get_input_cols(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def includes_preprocessing(obj) -> bool:
    if isinstance(obj, Pipeline):
        for name, step in obj.named_steps.items():
            if isinstance(step, ColumnTransformer) or "preprocess" in name.lower():
                return True
            if hasattr(step, "transformers"):
                return True
    return False

def maybe_set_output(obj):
    if obj is None or not hasattr(obj, "get_feature_names_out") or not hasattr(obj, "set_output"):
        return obj
    try:
        n = len(obj.get_feature_names_out())
    except Exception:
        return obj
    if n <= 300:
        try:
            obj.set_output(transform="pandas")
        except Exception:
            pass
    return obj

pre_cols = get_input_cols(pre)
sel_cols = get_input_cols(sel)
sel_wraps_pre = False
if isinstance(sel, Pipeline):
    if includes_preprocessing(sel):
        sel_wraps_pre = True
    elif pre_cols is not None and sel_cols is not None and list(sel_cols) == list(pre_cols):
        sel_wraps_pre = True

if sel_wraps_pre:
    frontend = maybe_set_output(sel)
    steps = [("selector_all", sel)]
else:
    frontend = maybe_set_output(pre)
    steps = [("preprocessor", pre)]
    if sel is not None:
        steps.append(("selector_all", sel))

for obj in (pre, sel):
    maybe_set_output(obj)

expected = get_input_cols(frontend)
if expected:
    missing = [c for c in expected if c not in feature_cols]
    if missing:
        raise ValueError(f"Saved frontend expects columns outside feature contract: {missing}")
    X = train[expected]
else:
    X = train[feature_cols]

y = train[targets]

rf = RandomForestRegressor(n_estimators=100, random_state=42)
pipe = Pipeline(steps + [("model", rf)])
fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(fitted, "alternative_model.pkl")
print("train rows", train.shape[0])
print("feature columns", len(feature_cols))
print("targets", targets)

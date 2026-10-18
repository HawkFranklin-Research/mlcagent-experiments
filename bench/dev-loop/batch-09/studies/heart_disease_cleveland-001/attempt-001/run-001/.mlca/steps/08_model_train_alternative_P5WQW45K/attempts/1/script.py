import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 42
mlca_runtime.set_seed(SEED)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
train = pd.read_parquet(inp["train"])
target = targets[0]
y = train[target]

def load(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

prev = load("07_model_train_logistic_J5YKBG0Y/model.pkl")
pre = load("05_preprocess_fit_preprocessing_G1EGCAW6/preprocessor.pkl")
sel = load("06_features_select_features_CZZQV6ME/selector.pkl")

def input_names(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        return [str(c) for c in obj.feature_names_in_]
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            names = input_names(step)
            if names is not None:
                return names
    return None

def key_like(s):
    if s.dtype.kind in "OUS" and s.nunique(dropna=False) == len(s):
        return True
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        return bool(s.is_monotonic_increasing and s.min() in (0, 1))
    return False

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
computed = [c for c in train.columns if c not in drop and not key_like(train[c])]

names = None
for obj in (prev, sel, pre):
    candidate = input_names(obj)
    if candidate and all(c in train.columns for c in candidate):
        names = candidate
        break

if names and all(c in train.columns for c in names):
    bad = set(names) & drop
    if bad:
        raise ValueError(f"Saved input names include forbidden columns: {sorted(bad)}")
    feature_cols = names
else:
    feature_cols = computed

X = train[feature_cols]
minority = y.value_counts(normalize=True).min()

def estimator():
    est = HistGradientBoostingClassifier(random_state=SEED)
    if minority < 0.10:
        est.set_params(class_weight="balanced")
    return est

if isinstance(prev, Pipeline) and len(prev.steps) >= 2:
    pipe = clone(prev)
    last = pipe.steps[-1][0]
    pipe.set_params(**{last: estimator()})
else:
    steps = []
    if isinstance(sel, Pipeline):
        has_pre = any(
            "preprocess" in name.lower() or isinstance(step, ColumnTransformer)
            for name, step in sel.steps
        )
        if has_pre:
            steps = list(sel.steps)
        else:
            if pre is not None:
                steps.append(("preprocessor", pre))
            steps.extend(sel.steps)
    else:
        if pre is not None:
            steps.append(("preprocessor", pre))
        if sel is not None:
            steps.append(("selector", sel))
    steps.append(("model", estimator()))
    pipe = Pipeline(steps)

try:
    pipe.set_output(transform="pandas")
except Exception:
    pass

pipe = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(pipe, "alternative_model.pkl")
print(f"train rows: {len(train)}")
print(f"features: {len(feature_cols)}")
print("saved alternative_model.pkl")

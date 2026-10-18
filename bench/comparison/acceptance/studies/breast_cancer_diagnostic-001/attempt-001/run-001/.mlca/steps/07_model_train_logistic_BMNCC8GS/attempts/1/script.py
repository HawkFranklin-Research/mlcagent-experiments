import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

def req(obj, key):
    if key not in obj:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {sorted(obj.keys())}")
    return obj[key]

protocol = json.loads(inputs["_protocol"].read_text())
targets = list(req(protocol, "targets"))
excluded = list(req(protocol, "excluded_columns"))
group = protocol.get("group_column")
task = req(protocol, "task")
_ = req(task, "primary_metric")

train = pd.read_parquet(inputs["train"])
target = targets[0]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inputs["05_preprocess_fit_preprocessing_WZN8YG48/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_select_features_6Y9H969T/selector.pkl"])

def is_key_like(s):
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)):
        return False
    if not s.notna().all() or s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_integer_dtype(s):
        v = s.sort_values().to_numpy()
        return len(v) > 1 and bool((v[1:] - v[:-1] == 1).all())
    return True

bad = set(targets) | set(excluded)
if group:
    bad.add(group)

raw_allowed = [
    c for c in train.columns
    if c not in bad and not is_key_like(train[c])
]

def raw_input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        names = list(obj.feature_names_in_)
        if all(n in train.columns for n in names) and not (set(names) & bad):
            return names
    return None

selector_full = (
    isinstance(sel, Pipeline)
    and len(sel.named_steps) > 1
    and any(hasattr(st, "transformers") or isinstance(st, Pipeline) for st in sel.named_steps.values())
)

feature_cols = raw_input_names(sel) if selector_full else None
feature_cols = feature_cols or raw_input_names(pre) or raw_allowed
feature_cols = list(feature_cols)

if set(feature_cols) & bad:
    raise ValueError("Target, excluded, or group column reached the feature set")

X = train[feature_cols]
y = train[target]

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
if selector_full:
    steps = list(sel.steps) + [("model", clf)]
else:
    steps = [("preprocessor", pre), ("selector", sel), ("model", clf)]

pipe = Pipeline(steps)
fitted = mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(fitted if fitted is not None else pipe, "model.pkl")

print(f"train_rows={len(train)} features={len(feature_cols)}")
print(f"frontend={'saved_selector' if selector_full else 'preprocessor+selector'}")
print("saved model.pkl")

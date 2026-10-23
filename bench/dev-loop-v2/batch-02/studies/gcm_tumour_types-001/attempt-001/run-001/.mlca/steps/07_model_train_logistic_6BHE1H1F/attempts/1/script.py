import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column") if "group_column" in protocol else None
train = pd.read_parquet(inputs["train"])

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_DXVQHGBE/preprocessor.pkl")
sel = load_pkl("06_features_select_features_9JJFXST3/selector.pkl")

def names_of(obj):
    try:
        return list(obj.feature_names_in_)
    except Exception:
        return None

pre_names = names_of(pre)
sel_names = names_of(sel)
drop = set(targets) | set(excluded) | ({group} if group else set())

base_cols = [c for c in train.columns if c not in drop]
keep = []
for c in base_cols:
    s = train[c]
    if s.nunique(dropna=False) == len(train) and (not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_integer_dtype(s)):
        continue
    keep.append(c)
base_cols = keep

if isinstance(sel, Pipeline) and len(sel.named_steps) > 1:
    frontend = sel
    expected = sel_names if sel_names is not None else pre_names
else:
    frontend = Pipeline([("preprocessor", pre), ("selector", sel)])
    expected = pre_names if pre_names is not None else base_cols

if expected is not None:
    bad = [c for c in expected if c in drop]
    if bad:
        raise ValueError(f"Leaked columns in expected features: {bad}")
    feature_cols = [c for c in expected if c in train.columns]
    missing = [c for c in expected if c not in train.columns]
    if missing:
        raise KeyError(f"Missing expected feature columns: {missing}")
else:
    feature_cols = base_cols

pipe = Pipeline([
    ("frontend", frontend),
    ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
])
X = train[feature_cols]
y = train[targets[0]]
mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(pipe, "model.pkl")
print(f"rows={len(train)} features={len(feature_cols)} classes={y.nunique()}")
print("saved model.pkl")

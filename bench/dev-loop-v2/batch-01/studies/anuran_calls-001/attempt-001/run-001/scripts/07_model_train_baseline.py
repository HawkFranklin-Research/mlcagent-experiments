import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing key {key!r}; available top-level keys: {sorted(d.keys())}")
    return d[key]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = need(protocol, "group_column")
task = need(protocol, "task")

train = pd.read_parquet(paths["train"])
pre = pickle.load(open(paths["05_preprocess_fit_preprocessing_A4M5DTDE/preprocessor.pkl"], "rb"))
sel = pickle.load(open(paths["06_features_retain_features_KKEEW75S/selector_all.pkl"], "rb"))

def input_names(obj):
    n = getattr(obj, "feature_names_in_", None)
    if n is not None:
        return list(n)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            n = getattr(step, "feature_names_in_", None)
            if n is not None:
                return list(n)
    return None

def is_selector_like(obj):
    name = obj.__class__.__name__
    return name.startswith("Select") or name in ("VarianceThreshold", "RFE", "RFECV")

def wraps_preprocessing(obj):
    return isinstance(obj, Pipeline) and obj.steps and not is_selector_like(obj.steps[0][1])

forbidden = set(targets) | set(excluded) | {group}

def is_key_like(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.nunique(dropna=False) == len(s) and s.name.lower().endswith("id")
    return False

base_features = [
    c for c in train.columns
    if c not in forbidden and not is_key_like(train[c])
]

pre_names = input_names(pre)
sel_names = input_names(sel)

if isinstance(sel, Pipeline) and wraps_preprocessing(sel) and sel_names is not None and set(sel_names).issubset(set(base_features)):
    frontend = sel
    raw_cols = sel_names
else:
    frontend = Pipeline([("preprocessor", pre), ("selector", sel)])
    raw_cols = pre_names if pre_names is not None else base_features

bad = [c for c in raw_cols if c in forbidden]
missing = [c for c in raw_cols if c not in train.columns]
if bad:
    raise ValueError(f"Saved feature contract includes forbidden columns: {bad}")
if missing:
    raise ValueError(f"Saved feature contract expects missing columns: {missing}")

if hasattr(pre, "set_output"):
    try:
        pre.set_output(transform="pandas")
    except Exception:
        pass

X = train[raw_cols]
y = train[targets]

model = Pipeline([
    ("features", frontend),
    ("model", MultiOutputClassifier(LogisticRegression(max_iter=2000))),
])

mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(model, "model.pkl")
print(f"train rows={len(train)} features={len(raw_cols)} targets={len(targets)}")
print("saved model.pkl")

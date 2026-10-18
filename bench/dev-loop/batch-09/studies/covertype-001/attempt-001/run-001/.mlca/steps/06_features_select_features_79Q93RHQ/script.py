import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

with open(paths["_protocol"], "r", encoding="utf-8") as f:
    protocol = json.load(f)
with open(paths["05_preprocess_fit_preprocessing_0FJ6AMWD/manifest.json"], "r", encoding="utf-8") as f:
    manifest = json.load(f)
with open(paths["05_preprocess_fit_preprocessing_0FJ6AMWD/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

def required(mapping, key):
    if key not in mapping:
        raise KeyError(f"Missing required key {key!r}; available keys: {list(mapping.keys())}")
    return mapping[key]

targets = required(protocol, "targets")
excluded = required(protocol, "excluded_columns")
task = required(protocol, "task")
metric = required(task, "primary_metric")
group = protocol.get("group_column")
train = pd.read_parquet(paths["train"])

forbidden = set(targets) | set(excluded)
if group:
    forbidden.add(group)

def is_keylike(s):
    if len(s) == 0 or s.nunique(dropna=False) != len(s):
        return False
    name = str(s.name).lower()
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s):
        return any(tok in name for tok in ("id", "code", "key", "index", "row"))
    return False

computed_cols = [c for c in train.columns if c not in forbidden and not is_keylike(train[c])]
input_cols = list(getattr(pre, "feature_names_in_", []))
if input_cols:
    missing = [c for c in input_cols if c not in train.columns]
    if missing:
        raise ValueError(f"Preprocessor input columns missing from train: {missing}")
    bad = [c for c in input_cols if c in forbidden]
    if bad:
        raise ValueError(f"Forbidden target/excluded columns in preprocessor input: {bad}")
    feature_cols = input_cols
else:
    feature_cols = computed_cols

X_train = train[feature_cols].copy()
y_train = train[targets[0]]

def output_width(transformer, X):
    if hasattr(transformer, "get_feature_names_out"):
        try:
            return len(transformer.get_feature_names_out())
        except Exception:
            pass
    try:
        return transformer.transform(X.head(min(5, len(X)))).shape[1]
    except Exception:
        return len(X.columns)

n_features = int(output_width(pre, X_train))
k = min(20, n_features)

selector = SelectKBest(score_func=mutual_info_classif, k=k)
if isinstance(pre, Pipeline) and any(isinstance(step, SelectKBest) for step in pre.named_steps.values()):
    model = pre
else:
    model = Pipeline([("preprocessor", pre), ("selector", selector)])

if n_features <= 300:
    for obj in (pre, model):
        if hasattr(obj, "set_output"):
            try:
                obj.set_output(transform="pandas")
            except Exception:
                pass

model = mlca_runtime.fit_record(model, X_train, "train", y_train)

selected = k
if isinstance(model, Pipeline):
    for step in model.named_steps.values():
        if isinstance(step, SelectKBest):
            try:
                selected = int(step.get_support().sum())
            except Exception:
                pass

mlca_runtime.record_metric("encoded_features", float(n_features), "train")
mlca_runtime.record_metric("selected_features", float(selected), "train")
mlca_runtime.save_model(model, "selector.pkl")

print(f"train rows={len(train)} raw_features={len(feature_cols)} encoded_features={n_features} k={k}")
print(f"metric={metric} selected={selected}")
print("saved selector.pkl")

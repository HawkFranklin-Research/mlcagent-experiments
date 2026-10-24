import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")

train = pd.read_parquet(paths["train"])

with open(paths["05_preprocess_fit_preprocessing_CJG8K77A/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(paths["06_features_retain_features_BQCB3256/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_names(obj):
    fn = getattr(obj, "feature_names_in_", None)
    return list(fn) if fn is not None else []

pre_in = input_names(pre)
if pre_in:
    feature_cols = [
        c for c in pre_in
        if c in train.columns and c not in set(targets) and c not in excluded and c != group
    ]
else:
    drop = set(targets) | excluded | ({group} if group else set())
    n = len(train)
    feature_cols = []
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        keylike = (
            (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s))
            and s.nunique(dropna=False) == n
        )
        if not keylike:
            feature_cols.append(c)

X = train[feature_cols].copy()
y = train[targets].copy()

try:
    pre_out = list(pre.get_feature_names_out())
except Exception:
    pre_out = []

if pre_out and len(pre_out) <= 300 and hasattr(pre, "set_output"):
    try:
        pre.set_output(transform="pandas")
    except Exception:
        pass

sel_in = input_names(sel)
if isinstance(sel, Pipeline) and sel_in and set(sel_in) == set(feature_cols):
    feature_pipe = sel
elif sel_in and pre_out and set(sel_in).issubset(set(pre_out)):
    feature_pipe = Pipeline([("preprocessor", pre), ("selector", sel)])
elif sel_in and set(sel_in) == set(feature_cols):
    feature_pipe = sel
else:
    feature_pipe = Pipeline([("preprocessor", pre), ("selector", sel)])

model = Pipeline([
    ("features", feature_pipe),
    ("classifier", MultiOutputClassifier(LogisticRegression(max_iter=2000))),
])

fitted = mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(fitted, "model.pkl")
print(f"train rows={len(X)} features={len(feature_cols)} targets={len(targets)}")
print("saved model.pkl")

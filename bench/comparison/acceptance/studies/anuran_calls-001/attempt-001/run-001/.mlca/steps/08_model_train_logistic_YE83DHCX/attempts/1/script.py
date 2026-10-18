import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())


def need(key):
    if key not in proto:
        raise KeyError(f"Missing {key!r}; available keys: {list(proto.keys())}")
    return proto[key]


targets = need("targets")
targets = list(targets) if isinstance(targets, (list, tuple)) else [targets]
excluded = need("excluded_columns")
excluded = list(excluded) if isinstance(excluded, (list, tuple)) else [excluded]
group = proto.get("group_column")

train = pd.read_parquet(inputs["train"])
with open(inputs["06_preprocess_fit_preprocessing_4H3435T3/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["07_features_select_features_D255Z34M/selector.pkl"], "rb") as f:
    sel = pickle.load(f)


def first_input_cols(obj):
    if obj is None:
        return None
    if isinstance(obj, Pipeline):
        if not obj.steps:
            return None
        return first_input_cols(obj.steps[0][1])
    cols = getattr(obj, "feature_names_in_", None)
    return list(cols) if cols is not None else None


bad_cols = set(targets) | set(excluded)
if group:
    bad_cols.add(group)

candidate = [c for c in train.columns if c not in bad_cols]
n = len(train)
drop_key = []
for c in candidate:
    s = train[c]
    lname = c.lower()
    if lname in ("recordid", "record_id", "id") or lname.endswith("_id"):
        drop_key.append(c)
        continue
    if s.nunique(dropna=True) == n:
        keylike = (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_integer_dtype(s)
        )
        if keylike:
            drop_key.append(c)
raw_feature_cols = [c for c in candidate if c not in set(drop_key)]

raw_set = set(train.columns)
sel_in = first_input_cols(sel)
sel_accepts_raw = bool(sel_in) and set(sel_in).issubset(raw_set)

if sel_accepts_raw:
    steps = list(sel.steps) if isinstance(sel, Pipeline) else [("selector", sel)]
    expected = sel_in
else:
    steps = []
    if pre is not None:
        steps.append(("preprocessor", pre))
    if sel is not None:
        steps.append(("selector", sel))
    expected = first_input_cols(pre)
    if pre is None and sel is not None:
        raise ValueError("Selector expects transformed features but no preprocessor is available")

if expected:
    bad = set(expected) & bad_cols
    if bad:
        raise ValueError(f"Saved frontend expects excluded columns: {sorted(bad)}")
    feature_cols = expected
else:
    feature_cols = raw_feature_cols

if not feature_cols:
    raise ValueError("No feature columns remain after protocol exclusions")

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
pipeline = Pipeline(steps + [("model", clf)])

X = train[feature_cols]
y = train[targets[0]]
model = mlca_runtime.fit_record(pipeline, X, "train", y)
mlca_runtime.save_model(model, "model.pkl")
print(f"train rows={len(train)} features={len(feature_cols)}")

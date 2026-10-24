import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def req(d, key):
    if key not in d:
        raise KeyError(f"missing {key}; available top-level keys: {list(d.keys())}")
    return d[key]

targets = req(protocol, "targets")
if isinstance(targets, str):
    targets = [targets]
if not targets:
    raise ValueError("protocol targets is empty")
task = req(protocol, "task")
group = protocol.get("group_column")

exc_raw = req(protocol, "excluded_columns")
if isinstance(exc_raw, dict):
    for key in ("columns", "all_columns", "excluded_columns"):
        if key in exc_raw:
            excluded = set(exc_raw[key])
            break
    else:
        raise ValueError(f"excluded_columns dict lacks a full column list; keys={list(exc_raw)}")
else:
    excluded = set(exc_raw)

train = pd.read_parquet(inputs["train"])

def raw_feature_columns(df, target_cols, excluded_cols, group_col):
    cols = []
    for c in df.columns:
        if c in target_cols or c in excluded_cols or c == group_col:
            continue
        s = df[c]
        n = s.nunique(dropna=False)
        if n == len(df):
            low = str(c).lower()
            if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
                    or isinstance(s.dtype, pd.CategoricalDtype)):
                continue
            if pd.api.types.is_integer_dtype(s) and any(k in low for k in ("id", "key", "code")):
                continue
        cols.append(c)
    return cols

raw_cols = raw_feature_columns(train, set(targets), excluded, group)

with open(inputs["05_preprocess_fit_preprocessing_WM2D925J/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_8X366KXZ/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_names(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        return [str(c) for c in obj.feature_names_in_]
    if isinstance(obj, Pipeline):
        for _, step in obj.named_steps.items():
            names = input_names(step)
            if names is not None:
                return names
    return None

def output_names(obj):
    if hasattr(obj, "get_feature_names_out"):
        try:
            return list(obj.get_feature_names_out())
        except Exception:
            return None
    return None

pre_in = input_names(pre)
sel_in = input_names(sel)
pre_out = output_names(pre)

if pre_in is not None:
    bad = set(pre_in) & (set(targets) | excluded | ({group} if group else set()))
    if bad or not set(pre_in).issubset(set(raw_cols)):
        raise ValueError(f"preprocessor input columns are not valid raw features: {sorted(bad)}")
    feature_cols = list(pre_in)
else:
    feature_cols = raw_cols

if pre_out is not None and len(pre_out) <= 300 and hasattr(pre, "set_output"):
    try:
        pre.set_output(transform="pandas")
    except Exception:
        pass

if sel_in is not None and pre_out is not None and set(sel_in).issubset(set(pre_out)):
    steps = [("preprocessor", pre), ("selector", sel)]
elif sel_in is not None and isinstance(sel, Pipeline) and len(sel.named_steps) >= 2 and set(sel_in).issubset(set(feature_cols)):
    steps = [("selector", sel)]
elif sel_in is not None and set(sel_in).issubset(set(feature_cols)):
    steps = [("features", FeatureUnion([("preprocessor", pre), ("selector", sel)]))]
else:
    steps = [("preprocessor", pre), ("selector", sel)]

model = Pipeline(steps + [("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))])
target = targets[0]
mlca_runtime.fit_record(model, train[feature_cols], "train", y=train[target])
mlca_runtime.save_model(model, "model.pkl")

print("train rows", train.shape[0], "feature columns", len(feature_cols))
print("preprocessor input", None if pre_in is None else len(pre_in), "selector input", None if sel_in is None else len(sel_in))
print("saved model.pkl")

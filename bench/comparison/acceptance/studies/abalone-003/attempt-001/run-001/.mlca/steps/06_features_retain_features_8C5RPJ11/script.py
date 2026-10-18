import json
import pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key!r}; available top-level keys: {list(protocol.keys())}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
task = protocol["task"]

train = pd.read_parquet(inputs["train"])
manifest = json.loads(inputs["05_preprocess_fit_preprocessing_QJRR3HF7/manifest.json"].read_text())
with open(inputs["05_preprocess_fit_preprocessing_QJRR3HF7/preprocessor.pkl"], "rb") as fh:
    pre = pickle.load(fh)

if hasattr(pre, "feature_names_in_") and pre.feature_names_in_ is not None:
    feature_cols = list(pre.feature_names_in_)
else:
    drop = set(targets) | set(excluded) | {group}
    feature_cols = []
    n = len(train)
    for col in train.columns:
        if col in drop:
            continue
        s = train[col]
        is_key = False
        if s.nunique(dropna=False) == n:
            if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
                is_key = True
            elif pd.api.types.is_integer_dtype(s):
                vals = sorted(s.dropna().unique())
                if len(vals) == n and vals == list(range(int(vals[0]), int(vals[0]) + n)):
                    is_key = True
        if not is_key:
            feature_cols.append(col)

for bad in set(targets) | set(excluded) | {group}:
    if bad in feature_cols:
        raise ValueError(f"Feature list contains non-feature column {bad!r}")

X = train.loc[:, feature_cols]
y = train[targets[0]] if len(targets) == 1 else None
if len(targets) == 1:
    score_func = f_regression if "regression" in str(task.get("type", "")).lower() else f_classif
    selector = SelectKBest(score_func=score_func, k="all")
else:
    selector = FunctionTransformer(validate=False)

if isinstance(pre, Pipeline):
    names = {name for name, _ in pre.steps}
    step_name = "selector"
    i = 2
    while step_name in names:
        step_name = f"selector_{i}"
        i += 1
    pipe = Pipeline(list(pre.steps) + [(step_name, selector)])
else:
    pipe = Pipeline([("preprocess", pre), ("selector", selector)])

width = None
try:
    if hasattr(pre, "get_feature_names_out"):
        width = len(pre.get_feature_names_out())
except Exception:
    width = None

if width is not None and width <= 300:
    try:
        pipe.set_output(transform="pandas")
    except Exception:
        pass

mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(pipe, "selector_all.pkl")
print(f"train_rows={len(train)} features={len(feature_cols)} width={width}")
print("saved selector_all.pkl")

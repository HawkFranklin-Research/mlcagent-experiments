import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

if "targets" not in protocol:
    raise ValueError(f"Missing targets; available keys: {list(protocol.keys())}")
targets = protocol["targets"]
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]

excluded = set(protocol.get("excluded_columns", []))
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_R5N312M1/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_select_features_JQQFHDQ8/selector.pkl"], "rb") as f:
    selector = pickle.load(f)


def input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        first = next(iter(obj.named_steps.values()))
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    return None


pre_names = input_names(preprocessor)
if pre_names is not None:
    feature_cols = pre_names
else:
    feature_cols = []
    for c in train.columns:
        if c == target or c in excluded or (group is not None and c == group):
            continue
        s = train[c]
        if (s.dtype == object or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
            continue
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s) and s.is_monotonic_increasing:
            continue
        feature_cols.append(c)

bad = [c for c in feature_cols if c == target or c in excluded or (group is not None and c == group)]
if bad:
    raise ValueError(f"Target/excluded/group columns present in feature list: {bad[:5]}")

sel_names = input_names(selector)
pre_out = None
if hasattr(preprocessor, "get_feature_names_out"):
    try:
        pre_out = list(preprocessor.get_feature_names_out())
    except Exception:
        pre_out = None

if sel_names is not None and pre_names is not None and sel_names == pre_names:
    frontend = selector
elif sel_names is not None and pre_out is not None and sel_names == pre_out:
    frontend = Pipeline([("preprocess", preprocessor), ("select", selector)])
else:
    frontend = Pipeline([("preprocess", preprocessor), ("select", selector)])

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
model = Pipeline([("frontend", frontend), ("model", clf)])

mlca_runtime.fit_record(model, train[feature_cols], "train", train[target])
mlca_runtime.save_model(model, "model.pkl")
print("fit complete; model.pkl saved")

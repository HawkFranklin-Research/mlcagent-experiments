import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

protocol = json.loads(paths["_protocol"].read_text())
if "targets" not in protocol:
    raise KeyError(f"protocol missing 'targets'; top-level keys: {list(protocol)}")
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(paths["train"])
y = train[target]

with open(paths["05_preprocess_fit_preprocessing_M7MKT30F/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(paths["06_features_select_features_18G3F1C0/selector.pkl"], "rb") as f:
    sel = pickle.load(f)


def input_cols(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        try:
            return list(obj.feature_names_in_)
        except Exception:
            pass
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            cols = input_cols(step)
            if cols is not None:
                return cols
    return None


def has_pre_step(obj):
    if not isinstance(obj, Pipeline):
        return False
    names = set(obj.named_steps)
    if names & {"preprocessor", "preprocess", "preprocessing", "prep", "column_transformer"}:
        return True
    for _, step in obj.steps:
        if step.__class__.__name__ in {
            "ColumnTransformer", "SimpleImputer", "OneHotEncoder",
            "StandardScaler", "MinMaxScaler", "OrdinalEncoder",
        }:
            return True
    return False


pre_cols = input_cols(pre)
sel_cols = input_cols(sel)
drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

if sel is not None and has_pre_step(sel):
    steps = [("frontend", sel)]
    cols = sel_cols or pre_cols
else:
    steps = [("preprocessor", pre)]
    if sel is not None:
        steps.append(("selector", sel))
    cols = pre_cols or sel_cols

if cols is None:
    cols = [c for c in train.columns if c not in drop]
else:
    cols = [c for c in cols if c not in drop]

missing = [c for c in cols if c not in train.columns]
if missing:
    raise KeyError(f"Missing feature columns {missing}; available examples: {list(train.columns)[:20]}")
if not cols:
    raise ValueError("No feature columns remain after exclusions")

X = train[list(cols)].copy()
model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
pipe = Pipeline(steps + [("model", model)])
fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(fitted, "model.pkl")
print("Saved model.pkl with steps: " + ", ".join(name for name, _ in fitted.steps))

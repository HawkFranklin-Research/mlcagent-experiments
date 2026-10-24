import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]

def is_keylike(s):
    if s.isna().any() or s.nunique() != len(s):
        return False
    if s.dtype == object:
        return True
    if pd.api.types.is_integer_dtype(s.dtype):
        return not set(s.unique()).issubset({0, 1})
    return False

feature_cols = [c for c in feature_cols if not is_keylike(train[c])]

with open(inputs["05_preprocess_fit_preprocessing_F5B5FT5N/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_select_features_ZZ9TRA69/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

pre_in = list(getattr(preprocessor, "feature_names_in_", []))
if pre_in:
    feature_cols = pre_in

raw_set = set(feature_cols)

def fits_raw(obj):
    names = list(getattr(obj, "feature_names_in_", []))
    return bool(names) and set(names) == raw_set

if fits_raw(selector):
    frontend = selector
elif fits_raw(preprocessor):
    try:
        pre_out = set(preprocessor.get_feature_names_out(feature_cols))
    except Exception:
        pre_out = None
    sel_in = set(getattr(selector, "feature_names_in_", []))
    if sel_in and pre_out and sel_in == pre_out:
        frontend = Pipeline([("preprocessor", preprocessor), ("selector", selector)])
    else:
        frontend = Pipeline([("preprocessor", preprocessor), ("selector", selector)])
else:
    frontend = Pipeline([("preprocessor", preprocessor), ("selector", selector)])

hgb = HistGradientBoostingClassifier(random_state=42)
steps = list(frontend.steps) if isinstance(frontend, Pipeline) else [("frontend", frontend)]
steps.append(("classifier", hgb))
full_pipeline = Pipeline(steps)

X_train = train[feature_cols].copy()
y_train = train[targets[0]].copy()

fitted = mlca_runtime.fit_record(full_pipeline, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

print(f"features: {len(feature_cols)}")
print(f"train rows: {len(X_train)}")

import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
seed = 42
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

model_obj = load_pkl("07_model_train_logistic_8ET5BK8M/model.pkl")

def is_key_like(s):
    if s.nunique(dropna=False) != len(s):
        return False
    return (pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype))

if hasattr(model_obj, "feature_names_in_"):
    feature_cols = [c for c in model_obj.feature_names_in_
                    if c in train.columns and c not in targets and c not in excluded and c != group]
else:
    candidates = [c for c in train.columns if c not in targets and c not in excluded and c != group]
    feature_cols = [c for c in candidates if not is_key_like(train[c])]

X = train[feature_cols]
y = train[targets[0]]

hgb = HistGradientBoostingClassifier(random_state=seed)
if y.value_counts(normalize=True).min() < 0.10 and "class_weight" in hgb.get_params():
    hgb.set_params(class_weight="balanced")

if isinstance(model_obj, Pipeline):
    steps = list(model_obj.steps)
    if hasattr(steps[-1][1], "predict"):
        steps[-1] = (steps[-1][0], hgb)
    else:
        steps.append(("classifier", hgb))
    alt = Pipeline(steps)
else:
    pre = load_pkl("05_preprocess_fit_preprocessing_02TSADV2/preprocessor.pkl")
    sel = load_pkl("06_features_select_features_BS9MRTV1/selector.pkl")
    if isinstance(sel, Pipeline):
        alt = Pipeline(list(sel.steps) + [("classifier", hgb)])
    else:
        alt = Pipeline([("preprocessor", pre), ("selector", sel), ("classifier", hgb)])

alt = mlca_runtime.fit_record(alt, X, "train", y=y)
mlca_runtime.save_model(alt, "alternative_model.pkl")
print("features", len(feature_cols), "train", X.shape)
print("saved alternative_model.pkl")

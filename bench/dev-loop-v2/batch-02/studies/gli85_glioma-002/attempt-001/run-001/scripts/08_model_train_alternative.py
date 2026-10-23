import mlca_runtime
import json, pickle, copy
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

try:
    targets = protocol["targets"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {sorted(protocol)}")
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
target = targets[0]
if target not in train.columns:
    raise KeyError(f"Target {target!r} not found in train columns")

with inputs["07_model_train_logistic_TAB66YEF/model.pkl"].open("rb") as f:
    base = pickle.load(f)

feature_cols = getattr(base, "feature_names_in_", None)
if feature_cols is None and hasattr(base, "steps"):
    feature_cols = getattr(base.steps[0][1], "feature_names_in_", None)

if feature_cols is None:
    drop = set(targets) | set(excluded)
    if group_col:
        drop.add(group_col)

    def key_like(s):
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s):
            return s.nunique(dropna=True) == len(s)
        return False

    feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]
else:
    feature_cols = list(feature_cols)

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in feature_cols if c in train.columns and c not in drop]
if not feature_cols:
    raise ValueError("No feature columns available")

if hasattr(base, "steps"):
    steps = [(name, copy.deepcopy(obj)) for name, obj in base.steps[:-1]]
    pipe = Pipeline(steps + [("classifier", HistGradientBoostingClassifier(random_state=seed))])
else:
    with inputs["05_preprocess_fit_preprocessing_M7MKT30F/preprocessor.pkl"].open("rb") as f:
        pre = pickle.load(f)
    with inputs["06_features_select_features_18G3F1C0/selector.pkl"].open("rb") as f:
        sel = pickle.load(f)
    pipe = Pipeline([
        ("preprocessor", copy.deepcopy(pre)),
        ("selector", copy.deepcopy(sel)),
        ("classifier", HistGradientBoostingClassifier(random_state=seed)),
    ])

X = train.loc[:, feature_cols]
y = train[target]
fitted = mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(fitted, "alternative_model.pkl")
print(f"train rows={len(train)} features={len(feature_cols)} target={target}")
print("saved alternative_model.pkl")

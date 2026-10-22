import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
seed = 42

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key '{key}'. Available keys: {list(protocol)}")

targets = protocol["targets"]
excluded = protocol.get("excluded_columns") or []
group = protocol.get("group_column")
task = protocol["task"]
train = pd.read_parquet(inputs["train"])

protected = set(targets) | set(excluded)
if group:
    protected.add(group)


def is_key_like(series):
    if series.name in protected:
        return False
    n = len(series)
    if series.nunique(dropna=True) < n:
        return False
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return True
    if pd.api.types.is_integer_dtype(series):
        vals = series.dropna().sort_values().to_numpy()
        if len(vals) == n and (np.all(np.diff(vals) == 1) or np.all(np.diff(vals) == 0)):
            return True
    return False


feature_cols = [
    c for c in train.columns
    if c not in protected and not is_key_like(train[c])
]

with open(inputs["05_preprocess_fit_preprocessing_YDX4CF7Z/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_select_features_FT85WVQ5/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

pre_names = list(getattr(preprocessor, "feature_names_in_", []))
sel_names = list(getattr(selector, "feature_names_in_", []))

use_selector_only = False
if sel_names and (set(sel_names) == set(feature_cols) or (pre_names and set(sel_names) == set(pre_names))):
    use_selector_only = True
    raw_cols = sel_names
elif pre_names:
    raw_cols = pre_names
else:
    raw_cols = feature_cols

bad = set(raw_cols) & protected
if bad:
    raise ValueError(f"Protected columns reached the model input: {sorted(bad)}")

X_train = train[raw_cols]
y_train = train[targets[0]]

hgb = HistGradientBoostingClassifier(random_state=seed)
if y_train.value_counts(normalize=True).min() < 0.1 and "class_weight" in hgb.get_params():
    hgb.set_params(class_weight="balanced")

if use_selector_only:
    pipe = Pipeline([
        ("frontend", selector),
        ("model", hgb),
    ])
else:
    steps = [("preprocessor", preprocessor)]
    if selector is not None:
        steps.append(("selector", selector))
    steps.append(("model", hgb))
    pipe = Pipeline(steps)

pipe = mlca_runtime.fit_record(pipe, X_train, "train", y_train)
mlca_runtime.save_model(pipe, "alternative_model.pkl")

print(f"train rows={len(train)} features={len(raw_cols)}")
print("saved alternative_model.pkl")

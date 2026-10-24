import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

seed = 42
mlca_runtime.set_seed(seed)

inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r}; available top-level keys: {list(d.keys())}")
    return d[key]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group_col = protocol.get("group_column")
task = need(protocol, "task")
target = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

with open(inp["08_model_train_logistic_7JTSZ4DZ/model.pkl"], "rb") as f:
    base_model = pickle.load(f)

# Determine the raw input columns expected by the saved pipeline.
if hasattr(base_model, "feature_names_in_"):
    feature_cols = list(base_model.feature_names_in_)
else:
    feature_cols = []
    for c in train.columns:
        if c in targets or c in excluded or c == group_col:
            continue
        s = train[c]
        if (s.dtype == object or str(s.dtype).startswith("string")) and s.nunique(dropna=True) == s.notna().sum():
            continue
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(train):
            continue
        feature_cols.append(c)

for c in feature_cols:
    if c in targets or c in excluded:
        raise ValueError(f"Feature column {c!r} is a target or excluded column")

X_train = train[feature_cols]
y_train = train[target]

hgb = HistGradientBoostingClassifier(random_state=seed)

if isinstance(base_model, Pipeline):
    alt = Pipeline(list(base_model.steps[:-1]) + [("model", hgb)])
else:
    with open(inp["06_preprocess_fit_preprocessing_6C1GR31N/preprocessor.pkl"], "rb") as f:
        preprocessor = pickle.load(f)
    with open(inp["07_features_select_features_2XVWGVRB/selector.pkl"], "rb") as f:
        selector = pickle.load(f)
    alt = Pipeline([("preprocessor", preprocessor), ("selector", selector), ("model", hgb)])

alt = mlca_runtime.fit_record(alt, X_train, "train", y=y_train)
mlca_runtime.save_model(alt, "alternative_model.pkl")

print("train rows:", len(train), "features:", len(feature_cols))
print("classes:", list(alt.classes_) if hasattr(alt, "classes_") else "n/a")

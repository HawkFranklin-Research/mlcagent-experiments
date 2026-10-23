import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

seed = 42
mlca_runtime.set_seed(seed)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
target = targets[0]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
train = pd.read_parquet(inp["train"])

with open(inp["07_model_train_logistic_N3PQQ99D/model.pkl"], "rb") as f:
    base = pickle.load(f)

if isinstance(base, Pipeline):
    if hasattr(base, "feature_names_in_"):
        feature_cols = list(base.feature_names_in_)
    else:
        feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
    steps = base.steps[:-1] + [("classifier", HistGradientBoostingClassifier(random_state=seed))]
    alt = Pipeline(steps)
else:
    with open(inp["05_preprocess_fit_preprocessing_0GNR753P/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inp["06_features_select_features_MJW9JFWH/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
    alt = Pipeline([
        ("preprocessor", pre),
        ("selector", sel),
        ("classifier", HistGradientBoostingClassifier(random_state=seed)),
    ])

for c in targets + excluded + ([group] if group else []):
    if c in feature_cols:
        raise ValueError(f"Leaked column in features: {c}")

X = train[feature_cols]
y = train[target]
alt_fitted = mlca_runtime.fit_record(alt, X, "train", y=y)
mlca_runtime.save_model(alt_fitted, "alternative_model.pkl")
print("train", X.shape, "features", len(feature_cols))

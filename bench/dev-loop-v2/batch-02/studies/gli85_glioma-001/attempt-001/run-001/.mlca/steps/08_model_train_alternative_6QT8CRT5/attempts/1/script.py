import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 42
mlca_runtime.set_seed(SEED)

inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
if len(targets) != 1:
    raise ValueError("Expected exactly one target column")

train = pd.read_parquet(inp["train"])
with open(inp["07_model_train_logistic_XYKSK7V0/model.pkl"], "rb") as f:
    saved = pickle.load(f)
if not isinstance(saved, Pipeline) or len(saved.steps) < 2:
    raise TypeError("Saved model is not a fitted Pipeline with preprocessing and an estimator")

y = train[targets[0]].copy()
kwargs = {"random_state": SEED}
if y.value_counts(normalize=True).min() < 0.10:
    kwargs["class_weight"] = "balanced"

steps = list(saved.steps[:-1])
alternative = Pipeline(steps + [("hgb", HistGradientBoostingClassifier(**kwargs))])

feats = None
if hasattr(alternative, "feature_names_in_"):
    feats = list(alternative.feature_names_in_)
elif hasattr(alternative.steps[0][1], "feature_names_in_"):
    feats = list(alternative.steps[0][1].feature_names_in_)
if feats is None:
    drop = set(targets) | set(excluded) | ({group} if group else set())
    feats = [c for c in train.columns if c not in drop]

missing = [c for c in feats if c not in train.columns]
if missing:
    raise KeyError(f"Missing feature columns in train: {missing[:5]}")

X = train.loc[:, feats].copy()
alternative = mlca_runtime.fit_record(alternative, X, "train", y=y)
mlca_runtime.save_model(alternative, "alternative_model.pkl")
print(f"train rows={len(X)} features={len(feats)} model=HistGradientBoostingClassifier")

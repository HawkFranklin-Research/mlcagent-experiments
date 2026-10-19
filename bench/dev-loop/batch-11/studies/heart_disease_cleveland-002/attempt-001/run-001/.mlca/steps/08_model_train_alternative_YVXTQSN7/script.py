import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
targets = proto["targets"]
excl = list(proto["excluded_columns"])
group = proto.get("group_column")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

with open(inp["07_model_train_logistic_ZJV6JEEV/model.pkl"], "rb") as f:
    prior = pickle.load(f)

if hasattr(prior, "feature_names_in_"):
    feats = list(prior.feature_names_in_)
else:
    feats = [c for c in train.columns if c not in targets and c not in excl and c != group]

y = train[targets[0]] if len(targets) == 1 else train[targets]
minority = pd.Series(y).value_counts(normalize=True).min() if len(targets) == 1 else 1.0
hgb = HistGradientBoostingClassifier(
    random_state=42,
    class_weight="balanced" if minority < 0.10 else None,
)

if isinstance(prior, Pipeline) and len(prior.steps) >= 1:
    steps = [(n, clone(e)) for n, e in prior.steps[:-1]]
    pipeline = Pipeline(steps + [("model", hgb)])
else:
    with open(inp["05_preprocess_fit_preprocessing_FQWKV8QW/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inp["06_features_select_features_6RNK7ZHK/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    pipeline = Pipeline([("preprocessor", clone(pre)), ("selector", clone(sel)), ("model", hgb)])

pipeline = mlca_runtime.fit_record(pipeline, train[feats], "train", y)
mlca_runtime.save_model(pipeline, "alternative_model.pkl")

print("train", train.shape)
print("features", len(feats))
print("steps", [n for n, _ in pipeline.steps])

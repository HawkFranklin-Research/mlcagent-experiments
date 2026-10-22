import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

seed = 42
mlca_runtime.set_seed(seed)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
if len(targets) != 1:
    raise ValueError(f"expected one target, got {targets}")
target = targets[0]

train = pd.read_parquet(paths["train"])
with open(paths["latest_model"], "rb") as f:
    base = pickle.load(f)
if not isinstance(base, Pipeline):
    raise TypeError("latest_model is not a sklearn Pipeline")

feat = getattr(base, "feature_names_in_", None)
if feat is None:
    drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
    feat = [c for c in train.columns if c not in drop]
else:
    feat = list(feat)

bad = (set(feat) & set(targets)) | (set(feat) & set(excluded))
if bad:
    raise ValueError(f"target/excluded in features: {sorted(bad)}")
if group_col and group_col in feat:
    raise ValueError(f"group column in features: {group_col}")

X = train[feat]
y = train[target]
small = y.value_counts(normalize=True).min() < 0.10
kw = {"random_state": seed}
if small:
    kw["class_weight"] = "balanced"
clf = HistGradientBoostingClassifier(**kw)

steps = [(name, clone(est)) for name, est in base.steps[:-1]]
pipe = Pipeline(steps + [("classifier", clf)])
fitted = mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(fitted, "alternative_model.pkl")
print("train", train.shape, "features", len(feat))
print("balanced_class_weight", small)

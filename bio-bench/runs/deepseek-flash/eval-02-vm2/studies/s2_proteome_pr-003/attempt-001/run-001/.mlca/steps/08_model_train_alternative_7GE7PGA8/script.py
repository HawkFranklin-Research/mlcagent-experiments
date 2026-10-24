import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
ins = mlca_runtime.inputs()
seed = 42

proto = json.loads(ins["_protocol"].read_text())
targets = proto["targets"]
excluded = list(proto["excluded_columns"])
group = proto.get("group_column")
train = pd.read_parquet(ins["train"])

with open(ins["latest_model"], "rb") as f:
    latest = pickle.load(f)

if isinstance(latest, Pipeline) and hasattr(latest, "feature_names_in_"):
    feature_names = list(latest.feature_names_in_)
    base_steps = list(latest.steps[:-1])
else:
    raise RuntimeError(
        f"latest_model is not a usable Pipeline with feature_names_in_; "
        f"type={type(latest).__name__}, steps={list(getattr(latest, 'named_steps', {}).keys())}"
    )

missing = [c for c in feature_names if c not in train.columns]
if missing:
    raise RuntimeError(f"missing features in train: {missing[:10]}")

target = targets[0]
y = train[target]
frac_min = float(y.value_counts(normalize=True).min())
cw = "balanced" if frac_min < 0.1 else None

clf = HistGradientBoostingClassifier(random_state=seed, class_weight=cw)
new_pipe = Pipeline(steps=base_steps + [("clf", clf)])

X = train[feature_names].copy()
mlca_runtime.fit_record(new_pipe, X, "train", y=y)
mlca_runtime.save_model(new_pipe, "alternative_model.pkl")

print(f"features={len(feature_names)} train_rows={len(train)}")
print(f"min_class_frac={frac_min:.4f} class_weight={cw}")
print(f"steps={[s[0] for s in new_pipe.steps]}")

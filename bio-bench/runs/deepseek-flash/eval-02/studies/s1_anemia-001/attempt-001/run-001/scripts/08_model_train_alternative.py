import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns") or []
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
tgt = targets[0] if isinstance(targets, (list, tuple)) else targets
y = train[tgt]

with open(inputs["07_model_train_logistic_Y2147J2H/model.pkl"], "rb") as f:
    base = pickle.load(f)

frac = y.value_counts(normalize=True)
hgb_kwargs = {"random_state": 42}
if frac.min() < 0.1:
    hgb_kwargs["class_weight"] = "balanced"
clf = HistGradientBoostingClassifier(**hgb_kwargs)

if isinstance(base, Pipeline) and len(base.steps) >= 2:
    feats = list(base.feature_names_in_) if hasattr(base, "feature_names_in_") else [
        c for c in train.columns
        if c not in set(targets) | set(excluded) | ({group} if group else set())
    ]
    pipe = Pipeline(base.steps[:-1] + [("classifier", clf)])
else:
    with open(inputs["05_preprocess_fit_preprocessing_PTKJZF9S/preprocessor.pkl"], "rb") as f:
        pp = pickle.load(f)
    with open(inputs["06_features_select_features_00QZZ2DE/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    feats = list(pp.feature_names_in_) if hasattr(pp, "feature_names_in_") else [
        c for c in train.columns
        if c not in set(targets) | set(excluded) | ({group} if group else set())
    ]
    steps = []
    sel_in = list(getattr(sel, "feature_names_in_", []) or [])
    if not (sel_in and set(sel_in) == set(feats) and len(sel_in) == len(feats)):
        steps.append(("preprocessor", pp))
    steps += [("selector", sel), ("classifier", clf)]
    pipe = Pipeline(steps)

X = train[feats].copy()
mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(pipe, "alternative_model.pkl")
print(f"rows={len(X)} cols={X.shape[1]} classes={y.nunique()} cw={hgb_kwargs.get('class_weight', 'none')}")

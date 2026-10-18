import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

SEED = 42
mlca_runtime.set_seed(SEED)

paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
metric = protocol["task"].get("primary_metric", "roc_auc")

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


candidates = [
    "latest_model",
    "07_model_train_logistic_QMMJ5EQH/model.pkl",
    "08_model_train_alternative_3P2DK38C/alternative_model.pkl",
]
pipeline = None
for key in candidates:
    if key in paths:
        obj = load_pickle(paths[key])
        if isinstance(obj, Pipeline) and hasattr(obj, "predict_proba"):
            pipeline = obj
            break
if pipeline is None:
    raise RuntimeError("No complete saved Pipeline with predict_proba found")

forbidden = set(targets) | set(excluded) | ({group} if group else set())
candidate_cols = [c for c in train.columns if c not in forbidden]
feature_cols = []
for col in candidate_cols:
    s = train[col]
    is_keylike = (
        s.nunique(dropna=True) == len(s)
        and (s.dtype == object or pd.api.types.is_integer_dtype(s))
    )
    if not is_keylike:
        feature_cols.append(col)

if hasattr(pipeline, "feature_names_in_"):
    pipeline_cols = list(pipeline.feature_names_in_)
    if set(pipeline_cols) != set(feature_cols):
        raise ValueError("Pipeline input columns differ from frozen feature columns")
    feature_cols = pipeline_cols

missing = [c for c in feature_cols if c not in val.columns]
if missing:
    raise KeyError(f"Missing validation feature columns: {missing}")

target = targets[0]
mask = val[target].notna()
X_val = val.loc[mask, feature_cols].copy()
y_raw = val.loc[mask, target]

classes = getattr(pipeline, "classes_", None)
if classes is None:
    final_step = pipeline.named_steps[list(pipeline.named_steps)[-1]]
    classes = getattr(final_step, "classes_", None)
if classes is None:
    raise RuntimeError("Pipeline has no classes_ attribute")

classes = list(classes)
if len(classes) == 2:
    y_val = (y_raw == classes[1]).astype(int)
    scoring = "roc_auc"
else:
    mapping = {c: i for i, c in enumerate(classes)}
    y_val = y_raw.map(mapping).astype(int)
    scoring = "roc_auc_ovr" if metric in ("roc_auc", "roc_auc_ovr") else metric

result = permutation_importance(
    pipeline,
    X_val,
    y_val,
    scoring=scoring,
    n_repeats=20,
    random_state=SEED,
    n_jobs=1,
)

imp = pd.DataFrame(
    {
        "feature": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_sd": result.importances_std,
    }
)
imp = imp.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
for col in ("importance_mean", "importance_sd"):
    imp[col] = imp[col].map(lambda x: float(f"{x:.4g}"))

mlca_runtime.save_table(
    imp,
    "interpretation.csv",
    caption="Permutation importance on validation rows using primary metric scoring.",
)

top_feature = str(imp.loc[0, "feature"])
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"validation rows: {len(X_val)}; features: {len(feature_cols)}; top: {top_feature}")

import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")
targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group_col = protocol.get("group_column")
task = protocol["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
print(f"train rows={len(train)} val rows={len(val)}")

def key_like(col, s):
    if s.dtype == object or str(s.dtype) in ("string", "category"):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) == len(s):
        name = str(col).lower()
        return any(t in name for t in ("id", "key", "code", "row", "index"))
    return False

drop = set(targets) | excluded
if group_col is not None:
    drop.add(group_col)
feature_cols = [c for c in train.columns if c not in drop and not key_like(c, train[c])]
print(f"features={len(feature_cols)}")

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def expected_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def accepts_raw(obj):
    cols = expected_cols(obj)
    return cols is not None and len(cols) == len(feature_cols) and set(cols) == set(feature_cols)

model = None
input_cols = None
for key in (
    "latest_model",
    "07_model_train_logistic_0MNK5WE5/model.pkl",
    "08_model_train_alternative_4KBDBVMX/alternative_model.pkl",
):
    if key in inputs:
        obj = load_pkl(inputs[key])
        cols = expected_cols(obj)
        if cols is not None and len(cols) == len(feature_cols) and set(cols) == set(feature_cols):
            model, input_cols = obj, cols
            break

if model is None:
    pre = load_pkl(inputs["05_preprocess_fit_preprocessing_TZWQXPN7/preprocessor.pkl"])
    sel = load_pkl(inputs["06_features_select_features_NTK41RPZ/selector.pkl"])
    est = load_pkl(inputs["07_model_train_logistic_0MNK5WE5/model.pkl"])
    if accepts_raw(sel):
        front = sel
    elif accepts_raw(pre):
        front = Pipeline([("preprocessor", pre), ("selector", sel)])
    else:
        raise RuntimeError("No saved pipeline accepts the raw feature columns")
    model = Pipeline([("frontend", front), ("model", est)])
    input_cols = feature_cols

target = targets[0]
if target not in val.columns:
    raise KeyError(f"Target {target!r} not found in validation table")
y_val = val[target]
X_val = val[input_cols]

if task.get("type") == "binary" or task.get("n_classes") == 2:
    scoring = "roc_auc"
else:
    scoring = "roc_auc_ovr"

res = permutation_importance(
    model,
    X_val,
    y_val,
    scoring=scoring,
    n_repeats=20,
    random_state=42,
    n_jobs=1,
)
imp = pd.DataFrame({
    "feature": input_cols,
    "importance_mean": res.importances_mean,
    "importance_sd": res.importances_std,
})
imp = imp.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
imp = imp.round(4)

mlca_runtime.save_table(imp, "interpretation.csv", caption="Permutation importance on validation rows, top 15 features")
top_feature = imp.iloc[0]["feature"]
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"top_feature={top_feature}")

import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

seed = 42
mlca_runtime.set_seed(seed)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ["task", "targets", "excluded_columns"]:
    if key not in protocol:
        raise KeyError(f"Missing _protocol key {key!r}; available keys: {list(protocol.keys())}")

task = protocol["task"]
targets = protocol["targets"]
if isinstance(targets, str):
    targets = [targets]
excluded = protocol["excluded_columns"] or []
if isinstance(excluded, str):
    excluded = [excluded]
group_col = protocol.get("group_column", None)
target_col = targets[0]

train = pd.read_parquet(inputs["train"])
if target_col not in train.columns:
    raise KeyError(f"Target {target_col!r} not in train columns")

with open(inputs["07_model_train_logistic_94MG7111/model.pkl"], "rb") as f:
    base_model = pickle.load(f)

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

derived = [c for c in train.columns if c not in drop]
feature_cols = []
for c in derived:
    s = train[c]
    is_text = s.dtype == object or str(s.dtype).startswith("string") or str(s.dtype) == "category"
    if is_text and s.nunique(dropna=True) == len(s):
        continue
    if (c.lower() in {"id", "index", "row_id", "rowid"} or c.lower().endswith("_id")) and s.nunique(dropna=True) == len(s):
        continue
    feature_cols.append(c)

if hasattr(base_model, "feature_names_in_"):
    model_cols = list(base_model.feature_names_in_)
    missing = [c for c in model_cols if c not in train.columns]
    if missing:
        raise KeyError(f"Missing model input columns: {missing}")
    bad = set(model_cols) & (set(targets) | set(excluded) | ({group_col} if group_col else set()))
    if bad:
        raise ValueError(f"Target/excluded/group columns in model input: {bad}")
    feature_cols = model_cols

if isinstance(base_model, Pipeline) and len(base_model.steps) >= 2:
    last_name = base_model.steps[-1][0]
    new_pipeline = Pipeline(
        base_model.steps[:-1]
        + [(last_name, HistGradientBoostingClassifier(random_state=seed))]
    )
else:
    pre = pickle.load(open(inputs["05_preprocess_fit_preprocessing_6Y57Q816/preprocessor.pkl"], "rb"))
    sel = pickle.load(open(inputs["06_features_select_features_HZT7Z0Q4/selector.pkl"], "rb"))
    new_pipeline = Pipeline([
        ("preprocessor", pre),
        ("selector", sel),
        ("classifier", HistGradientBoostingClassifier(random_state=seed)),
    ])

X_train = train.loc[:, feature_cols]
y_train = train[target_col]
fitted = mlca_runtime.fit_record(new_pipeline, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

base_names = [name for name, _ in base_model.steps] if isinstance(base_model, Pipeline) else [type(base_model).__name__]
print(f"target={target_col} type={task.get('type')} metric={task.get('primary_metric')}")
print(f"train_rows={len(train)} features={len(feature_cols)}")
print(f"base_steps={base_names}")
print(f"new_final={fitted.steps[-1][0]}:{type(fitted.steps[-1][1]).__name__}")
print("saved alternative_model.pkl")

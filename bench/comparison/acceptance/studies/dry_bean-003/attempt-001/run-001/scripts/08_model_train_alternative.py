import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = list(protocol.get("excluded_columns") or [])
group = protocol.get("group_column")
task = protocol["task"]
primary_metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_keylike(s):
    n = int(s.notna().sum())
    u = int(s.nunique(dropna=True))
    if n == 0 or u <= 1:
        return False
    if pd.api.types.is_integer_dtype(s.dtype):
        return u == len(s)
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
        return u == n
    return False

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop and not is_keylike(train[c])]

y_train = train[targets[0]] if len(targets) == 1 else train[targets]
y_val = val[targets[0]] if len(targets) == 1 else val[targets]

with open(inputs["05_preprocess_fit_preprocessing_DYJSGEW8/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_select_features_KAD36QCW/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

def in_names(obj):
    return list(obj.feature_names_in_) if hasattr(obj, "feature_names_in_") else None

clf = HistGradientBoostingClassifier(random_state=42)
sel_in = in_names(selector)
raw_set = set(feature_cols)

if sel_in is not None and set(sel_in).issubset(raw_set):
    input_cols = sel_in
    steps = [("selector", selector), ("clf", clf)]
else:
    pre_in = in_names(preprocessor)
    input_cols = pre_in if pre_in is not None else feature_cols
    steps = [("preprocessor", preprocessor), ("selector", selector), ("clf", clf)]

missing = [c for c in input_cols if c not in train.columns]
if missing:
    raise ValueError(f"Missing input columns: {missing}")
X_train = train[input_cols].copy()
X_val = val[input_cols].copy()

pipeline = Pipeline(steps)
mlca_runtime.fit_record(pipeline, X_train, "train", y_train)
mlca_runtime.save_model(pipeline, "alternative_model.pkl")

if primary_metric == "roc_auc_ovr":
    proba = pipeline.predict_proba(X_val)
    score = float(roc_auc_score(y_val, proba, multi_class="ovr",
                                average="macro", labels=list(pipeline.classes_)))
    mlca_runtime.record_metric("roc_auc_ovr", score, "val")
    print("val roc_auc_ovr:", round(score, 4))
print("n_features:", len(input_cols), "n_train:", len(X_train))

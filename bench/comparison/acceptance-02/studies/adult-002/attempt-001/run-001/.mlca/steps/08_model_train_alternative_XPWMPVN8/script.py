import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SEED = 42
mlca_runtime.set_seed(SEED)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {list(protocol)}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
task = protocol["task"]
if len(targets) != 1:
    raise ValueError(f"Expected one target, found {targets}")
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_pkl(inputs["latest_model"])

features = None
if isinstance(model, Pipeline):
    try:
        fn = list(model.feature_names_in_)
    except Exception:
        fn = None
    if fn:
        features = fn

if features is None:
    features = [c for c in train.columns if c not in targets and c not in excluded and c != group]
    keep = []
    n = len(train)
    for c in features:
        s = train[c]
        if s.nunique(dropna=False) == n:
            if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
                continue
            if pd.api.types.is_integer_dtype(s) and s.is_monotonic_increasing:
                continue
        keep.append(c)
    features = keep

bad = [c for c in features if c in targets or c in excluded or c == group]
if bad:
    raise ValueError(f"Features include target/excluded/group columns: {bad}")
missing = [c for c in features if c not in train.columns or c not in val.columns]
if missing:
    raise KeyError(f"Missing feature columns: {missing}")

X_train = train[features]
X_val = val[features]
y_train = train[target]
y_val = val[target]

hgb = HistGradientBoostingClassifier(random_state=SEED)

if isinstance(model, Pipeline):
    new_steps = list(model.steps)[:-1] + [("hist_gradient_boosting", hgb)]
    new_model = Pipeline(new_steps)
else:
    pre = load_pkl(inputs["05_preprocess_fit_preprocessing_WBRZJAT5/preprocessor.pkl"])
    sel = load_pkl(inputs["06_features_select_features_PDWSHJGW/selector.pkl"])
    if isinstance(sel, Pipeline) and any("preprocess" in n.lower() for n in sel.named_steps):
        new_model = Pipeline(list(sel.steps) + [("hist_gradient_boosting", hgb)])
    else:
        new_model = Pipeline([("preprocessor", pre), ("selector", sel), ("hist_gradient_boosting", hgb)])

fitted = mlca_runtime.fit_record(new_model, X_train, "train", y_train)
new_model = fitted if fitted is not None else new_model

classes = list(new_model.classes_)
if task.get("primary_metric") == "roc_auc" and len(classes) == 2:
    pos = task.get("positive_label", classes[1])
    if pos not in classes:
        pos = classes[1]
    idx = classes.index(pos)
    proba = new_model.predict_proba(X_val)
    auc = roc_auc_score((y_val == pos).astype(int), proba[:, idx])
    mlca_runtime.record_metric("val_roc_auc", float(auc), "val")
    print(f"features={len(features)} train_rows={len(X_train)} val_rows={len(X_val)}")
    print(f"val_roc_auc={auc:.4f}")

mlca_runtime.save_model(new_model, "alternative_model.pkl")

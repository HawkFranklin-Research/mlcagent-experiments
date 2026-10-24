import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets, excluded = protocol["targets"], protocol["excluded_columns"]
group_col = protocol.get("group_column")
task = protocol["task"]
pos_label = task.get("positive_label", protocol.get("positive_label"))

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
drop_cols = set(targets) | set(excluded) | ({group_col} if group_col else set())
feat_cols = []
for c in train.columns:
    if c in drop_cols:
        continue
    s = train[c]; nun = s.nunique(dropna=True)
    is_str = s.dtype == object or pd.api.types.is_string_dtype(s)
    if is_str and nun == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and nun == len(s) and not set(s.dropna().unique()).issubset({0, 1}):
        continue
    feat_cols.append(c)

with open(inputs["05_preprocess_fit_preprocessing_T6A9FCB5/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_WE31FN6Z/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def _has_pre(obj):
    if isinstance(obj, Pipeline):
        return any(("pre" in n.lower() or "column" in n.lower() or "transformer" in n.lower())
                   for n in obj.named_steps)
    return False

tgt = targets[0]
y_train, y_val = train[tgt], val[tgt]
cw = "balanced" if y_train.value_counts(normalize=True).min() < 0.10 else None
if cw:
    try:
        hgb = HistGradientBoostingClassifier(random_state=42, class_weight=cw)
    except TypeError:
        hgb = HistGradientBoostingClassifier(random_state=42)
else:
    hgb = HistGradientBoostingClassifier(random_state=42)

if _has_pre(sel):
    front_cols = list(sel.feature_names_in_) if hasattr(sel, "feature_names_in_") else feat_cols
    pipe = Pipeline([("selector", sel), ("hgb", hgb)])
else:
    front_cols = list(pre.feature_names_in_) if hasattr(pre, "feature_names_in_") else feat_cols
    pipe = Pipeline([("preprocessor", pre), ("selector", sel), ("hgb", hgb)])

X_train, X_val = train[front_cols], val[front_cols]
mlca_runtime.fit_record(pipe, X_train, partition="train", y=y_train)

classes = list(pipe.classes_)
idx = classes.index(pos_label) if pos_label in classes else 1
proba = pipe.predict_proba(X_val)[:, idx]
auc = roc_auc_score((y_val.values == classes[idx]).astype(int), proba)
mlca_runtime.record_metric("val_roc_auc_alternative_hgb", float(auc), split="val")
mlca_runtime.save_model(pipe, "alternative_model.pkl")
print(f"features={len(front_cols)} train={len(X_train)} val_auc={auc:.4f}")

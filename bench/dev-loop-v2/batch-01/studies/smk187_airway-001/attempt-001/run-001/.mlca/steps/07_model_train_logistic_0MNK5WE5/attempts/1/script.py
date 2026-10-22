import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

train = pd.read_parquet(paths["train"])
protocol = json.loads(paths["_protocol"].read_text())

targets = protocol.get("targets") or protocol["task"]["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)
target = targets[0]

with open(paths["05_preprocess_fit_preprocessing_TZWQXPN7/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(paths["06_features_select_features_NTK41RPZ/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

drop_cols = set(targets) | set(excluded)
if group:
    drop_cols.add(group)

def base_feature_columns(df):
    return [c for c in df.columns if c not in drop_cols]

raw_cols = None
if isinstance(sel, Pipeline) and any(
    name in ("preprocessor", "preprocess") or step is pre
    for name, step in sel.named_steps.items()
):
    raw_cols = getattr(sel, "feature_names_in_", None)
if raw_cols is None and hasattr(pre, "feature_names_in_"):
    raw_cols = list(pre.feature_names_in_)
if raw_cols is None:
    raw_cols = base_feature_columns(train)

raw_cols = [c for c in raw_cols if c in train.columns and c not in drop_cols]

sel_has_pre = isinstance(sel, Pipeline) and any(
    name in ("preprocessor", "preprocess") or step is pre
    for name, step in sel.named_steps.items()
)
sel_expects_raw = (
    hasattr(sel, "feature_names_in_")
    and list(sel.feature_names_in_) == list(raw_cols)
)

if sel_has_pre or sel_expects_raw:
    frontend = sel
else:
    frontend = Pipeline([("preprocessor", pre), ("selector", sel)])

model = Pipeline([
    ("frontend", frontend),
    ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
])

X = train[raw_cols]
y = train[target]

mlca_runtime.fit_record(model, X, "train", y)
mlca_runtime.save_model(model, "model.pkl")

print("train rows:", len(train))
print("feature columns:", len(raw_cols))
print("target:", target, "classes:", train[target].nunique())

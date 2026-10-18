import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available: {list(protocol.keys())}")
if "primary_metric" not in protocol["task"]:
    raise KeyError(f"Missing task key 'primary_metric'; available: {list(protocol['task'].keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(paths["train"])
with open(paths["05_preprocess_fit_preprocessing_S0BP8ZJM/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(paths["06_features_select_features_R97WA2VA/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def unwrap(obj):
    if isinstance(obj, dict):
        for k in ("preprocessor", "pipeline", "selector", "model"):
            if k in obj:
                return obj[k]
    return obj

pre = unwrap(pre)
sel = unwrap(sel)

def is_key_like(s):
    n = len(s)
    if n == 0 or s.nunique(dropna=False) != n:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s):
        d = s.sort_values().diff().dropna()
        return d.nunique() == 1 and (s.max() - s.min() + 1) == n
    return False

blocked = set(targets) | set(excluded) | ({group_col} if group_col else set())
candidates = [
    c for c in train.columns
    if c not in blocked and not is_key_like(train[c])
]

if hasattr(pre, "feature_names_in_"):
    pre_cols = list(pre.feature_names_in_)
    if set(pre_cols) & blocked:
        raise ValueError("Saved preprocessor expects a target or excluded column")
    feature_cols = pre_cols
else:
    feature_cols = candidates

def has_selector(obj):
    if not isinstance(obj, Pipeline):
        return False
    for name, step in obj.named_steps.items():
        if "select" in name.lower() or "select" in step.__class__.__name__.lower():
            return True
    return False

def has_preprocessing(obj):
    if not isinstance(obj, Pipeline):
        return False
    for name, step in obj.named_steps.items():
        if "preprocess" in name.lower() or isinstance(step, ColumnTransformer):
            return True
    return False

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

if sel is not None and has_preprocessing(sel):
    if hasattr(sel, "feature_names_in_"):
        sel_cols = list(sel.feature_names_in_)
        if not (set(sel_cols) & blocked):
            feature_cols = sel_cols
    steps = list(sel.steps) + [("classifier", clf)] if isinstance(sel, Pipeline) else [("selector", sel), ("classifier", clf)]
elif has_selector(pre):
    steps = list(pre.steps) + [("classifier", clf)] if isinstance(pre, Pipeline) else [("preprocessor", pre), ("classifier", clf)]
elif sel is not None:
    steps = [("preprocessor", pre), ("selector", sel), ("classifier", clf)]
else:
    steps = [("preprocessor", pre), ("classifier", clf)]

pipeline = Pipeline(steps)
X = train[feature_cols]
y = train[target]

fitted = mlca_runtime.fit_record(pipeline, X, "train", y=y)
model = fitted if fitted is not None else pipeline
mlca_runtime.save_model(model, "model.pkl")

print(f"train_rows={len(train)} features={len(feature_cols)} target={target}")
print(f"steps={[name for name, _ in model.steps]}")

import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

def req(d, k):
    if k not in d:
        raise RuntimeError(f"Missing {k}; available top-level keys: {list(d)}")
    return d[k]

protocol = json.loads(inp["_protocol"].read_text())
targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
task = req(protocol, "task")
group_col = protocol.get("group_column", None)
if task.get("type") != "multiclass":
    raise ValueError(f"Expected multiclass task, got {task.get('type')}")
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")

train = pd.read_parquet(inp["train"])
with open(inp["05_preprocess_fit_preprocessing_0GNR753P/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_select_features_MJW9JFWH/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

target = targets[0]
if hasattr(pre, "feature_names_in_"):
    raw_cols = list(pre.feature_names_in_)
else:
    raw_cols = [c for c in train.columns if c not in set(targets) | set(excluded) and c != group_col]

missing = [c for c in raw_cols if c not in train.columns]
if missing:
    raise KeyError(f"Missing feature columns, first few: {missing[:5]}")

forbidden = set(raw_cols) & (set(targets) | set(excluded) | ({group_col} if group_col else set()))
if forbidden:
    raise ValueError(f"Forbidden columns in feature set: {sorted(forbidden)}")

X = train.loc[:, raw_cols].copy()
y = train[target].copy()

model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

def selector_includes_preprocessor(s, p, raw):
    if not isinstance(s, Pipeline):
        return False
    for step in s.named_steps.values():
        if step is p:
            return True
    first = list(s.named_steps.values())[0]
    cols = getattr(first, "feature_names_in_", None)
    if cols is not None and list(cols) == raw:
        return True
    return isinstance(first, (ColumnTransformer, Pipeline))

if selector_includes_preprocessor(sel, pre, raw_cols):
    final = Pipeline(list(sel.steps) + [("clf", model)])
else:
    final = Pipeline([("preprocessor", pre), ("selector", sel), ("clf", model)])

fitted = mlca_runtime.fit_record(final, X, "train", y=y)
mlca_runtime.save_model(fitted, "model.pkl")

print(f"train rows={len(X)} features={len(raw_cols)} classes={y.nunique()}")
print(f"pipeline steps={list(fitted.named_steps)}")

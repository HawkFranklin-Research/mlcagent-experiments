import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

def need(k):
    if k not in proto:
        raise KeyError(f"Missing protocol key {k!r}; available: {sorted(proto)}")

for k in ("targets", "excluded_columns", "task"):
    need(k)

targets = proto["targets"]
if not isinstance(targets, list) or len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets!r}")
target = targets[0]

excluded = proto["excluded_columns"]
if isinstance(excluded, dict):
    excluded = excluded.get("columns", excluded.get("examples", []))
excluded = set(excluded) if isinstance(excluded, (list, tuple, set)) else set()
group = proto.get("group_column", None)

train = pd.read_parquet(inp["train"])
with open(inp["05_preprocess_fit_preprocessing_R1GAM795/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_select_features_FJMHG7YC/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

includes_pre = False
if isinstance(sel, Pipeline) and len(sel.steps) > 1:
    first = sel.steps[0][1]
    if hasattr(first, "feature_names_in_") and hasattr(pre, "feature_names_in_"):
        try:
            includes_pre = np.array_equal(first.feature_names_in_, pre.feature_names_in_)
        except Exception:
            includes_pre = False
if not includes_pre and hasattr(sel, "feature_names_in_") and hasattr(pre, "feature_names_in_"):
    try:
        includes_pre = np.array_equal(sel.feature_names_in_, pre.feature_names_in_)
    except Exception:
        includes_pre = False

frontend = sel if includes_pre else Pipeline([("preprocessor", pre), ("selector", sel)])

if hasattr(frontend, "feature_names_in_"):
    raw_cols = list(frontend.feature_names_in_)
elif hasattr(pre, "feature_names_in_"):
    raw_cols = list(pre.feature_names_in_)
elif hasattr(sel, "feature_names_in_"):
    raw_cols = list(sel.feature_names_in_)
else:
    raw_cols = [
        c for c in train.columns
        if c not in set(targets) and c not in excluded and c != group
    ]

if target in raw_cols:
    raise ValueError("Target column reached the feature input")
missing = [c for c in raw_cols if c not in train.columns]
if missing:
    raise KeyError(f"Missing feature columns: {missing[:5]} ...")

X = train[raw_cols]
y = train[target]

pipeline = Pipeline([
    ("frontend", frontend),
    ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
])
fitted = mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(fitted, "model.pkl")

print(f"train={train.shape} features={len(raw_cols)} target={target}")
print("saved model.pkl")

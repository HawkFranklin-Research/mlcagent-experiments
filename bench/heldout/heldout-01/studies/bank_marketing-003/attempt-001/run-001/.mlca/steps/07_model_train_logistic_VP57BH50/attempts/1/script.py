import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

protocol = json.loads(inp["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing_keys = [k for k in required if k not in protocol]
if missing_keys:
    raise KeyError(f"Missing protocol keys {missing_keys}; available top-level keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]
excluded = set(protocol["excluded_columns"])
group_col = protocol["group_column"]
bad = excluded | {target}
if group_col:
    bad.add(group_col)

train = pd.read_parquet(inp["train"])
if target not in train.columns:
    raise ValueError(f"Target {target!r} not in train columns")

with open(inp["05_preprocess_fit_preprocessing_KGJNVG5D/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_select_features_20NTEHHR/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_cols(obj):
    cols = getattr(obj, "feature_names_in_", None)
    return list(cols) if cols is not None else []

pre_cols = input_cols(pre)
sel_cols = input_cols(sel)
use_selector_only = bool(sel_cols and (not pre_cols or set(sel_cols) == set(pre_cols)))

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

if use_selector_only:
    pipeline = Pipeline([("selector", sel), ("model", clf)])
    frontend = sel
else:
    steps = []
    if pre is not None:
        steps.append(("preprocessor", pre))
    if sel is not None:
        steps.append(("selector", sel))
    steps.append(("model", clf))
    pipeline = Pipeline(steps)
    frontend = pre if pre is not None else sel

    if pre is not None and hasattr(pre, "get_feature_names_out"):
        n_out = len(pre.get_feature_names_out())
        if n_out <= 300 and hasattr(pre, "set_output"):
            pre.set_output(transform="pandas")
            if hasattr(pipeline, "set_output"):
                pipeline.set_output(transform="pandas")
        elif n_out > 300 and sel_cols and pre_cols and set(sel_cols) != set(pre_cols):
            raise RuntimeError("Wide preprocessor output cannot feed a name-based selector")

raw_cols = input_cols(frontend)
if raw_cols:
    overlap = [c for c in raw_cols if c in bad]
    if overlap:
        raise ValueError(f"Frontend expects excluded columns: {overlap}")
else:
    raw_cols = [c for c in train.columns if c not in bad]
    n = len(train)
    keep = []
    for c in raw_cols:
        s = train[c]
        is_text = (s.dtype == object or str(s.dtype).startswith("string"))
        if is_text and n > 1 and s.nunique(dropna=False) == n:
            continue
        keep.append(c)
    raw_cols = keep

missing = [c for c in raw_cols if c not in train.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

X_train = train[raw_cols]
y_train = train[target]
pipeline = mlca_runtime.fit_record(pipeline, X_train, "train", y_train)
mlca_runtime.save_model(pipeline, "model.pkl")

print(f"train_rows={len(train)} features={len(raw_cols)}")
print(f"steps={[name for name, _ in pipeline.steps]}")
print("saved model.pkl")

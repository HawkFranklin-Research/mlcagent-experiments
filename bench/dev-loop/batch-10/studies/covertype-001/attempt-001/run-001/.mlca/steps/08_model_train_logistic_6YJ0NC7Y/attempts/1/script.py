import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
protocol = json.loads(inp["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
task = protocol["task"]
if len(targets) != 1:
    raise RuntimeError(f"Expected one target, got {targets}")
target = targets[0]

def is_key_like(s, name):
    n = len(s)
    if s.nunique(dropna=False) < n:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        vals = pd.Series(s.dropna().unique()).sort_values().to_numpy()
        if len(vals) == n and n > 1 and (vals[-1] - vals[0] + 1) == n:
            return True
    return False

raw_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    if is_key_like(train[c], c):
        continue
    raw_cols.append(c)

def first_input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def has_selector(obj):
    if not isinstance(obj, Pipeline):
        return False
    return any(
        "select" in name.lower() or hasattr(step, "get_support")
        for name, step in obj.named_steps.items()
    )

def accepts_raw(obj):
    names = first_input_names(obj)
    return names is not None and set(names) == set(raw_cols) and len(names) == len(raw_cols)

with open(inp["06_preprocess_fit_preprocessing_XZ1WZNJH/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["07_features_select_features_KRQ7615Z/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

if isinstance(sel, Pipeline) and accepts_raw(sel) and has_selector(sel):
    frontend = sel
    model = Pipeline([("frontend", frontend), ("clf", clf)])
    input_cols = first_input_names(frontend)
elif accepts_raw(pre):
    frontend = pre
    steps = [("preprocessor", frontend)]
    if not has_selector(pre):
        steps.append(("selector", sel))
    steps.append(("clf", clf))
    model = Pipeline(steps)
    input_cols = first_input_names(frontend)
elif accepts_raw(sel):
    frontend = sel
    model = Pipeline([("frontend", frontend), ("clf", clf)])
    input_cols = first_input_names(frontend)
else:
    raise RuntimeError(
        "Could not find a saved frontend that accepts the raw feature schema; "
        f"pre names={first_input_names(pre)}, sel names={first_input_names(sel)}, raw={raw_cols}"
    )

if input_cols is None:
    input_cols = raw_cols
bad = set(targets) | set(excluded) | ({group} if group else set())
if bad.intersection(input_cols):
    raise RuntimeError(f"Target/excluded/group columns reached features: {sorted(bad.intersection(input_cols))}")

X = train[input_cols]
y = train[target]
mlca_runtime.fit_record(model, X, "train", y)
mlca_runtime.save_model(model, "model.pkl")
print(f"train_rows={len(train)} features={len(input_cols)} classes={y.nunique()}")

import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
train = pd.read_parquet(inp["train"])

with open(inp["05_preprocess_fit_preprocessing_DPM8GJVN/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_6T0PA2SA/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def manifest_cols(path):
    m = json.loads(path.read_text())
    for k in ("feature_columns", "features", "input_columns", "columns"):
        if k in m and isinstance(m[k], list):
            return m[k]
    return None

def keylike(s):
    s = s.dropna()
    if len(s) == 0:
        return False
    if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
        return s.nunique() == len(s) and s.nunique() > max(20, 0.5 * len(s))
    if pd.api.types.is_integer_dtype(s.dtype):
        u = sorted(s.unique())
        return len(u) == len(s) and len(u) > 50 and all(u[i+1] - u[i] == 1 for i in range(len(u)-1))
    return False

declared = manifest_cols(inp["05_preprocess_fit_preprocessing_DPM8GJVN/manifest.json"])
if declared and all(c in train.columns for c in declared):
    features = [c for c in declared if c not in targets and c not in excluded and c != group_col]
else:
    features = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
    features = [c for c in features if not keylike(train[c])]
if not features:
    raise ValueError("No feature columns remain after exclusions")

for obj in (pre, sel):
    if hasattr(obj, "set_output"):
        try:
            obj.set_output(transform="pandas")
        except Exception:
            pass

def input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for st in obj.named_steps.values():
            if hasattr(st, "feature_names_in_"):
                return list(st.feature_names_in_)
    return None

def accepts_raw(obj, cols):
    ic = input_cols(obj)
    if ic is None:
        return False
    return set(ic) == set(cols) and len(ic) == len(cols) and not (set(ic) & set(targets))

clf = MultiOutputClassifier(LogisticRegression(max_iter=2000))
if accepts_raw(sel, features):
    features = list(sel.feature_names_in_)
    pipe = Pipeline([("selector", sel), ("clf", clf)])
elif accepts_raw(pre, features):
    pipe = Pipeline([("preprocessor", pre), ("selector", sel), ("clf", clf)])
else:
    raise ValueError("Saved preprocessor/selector do not accept the declared raw feature columns")

model = mlca_runtime.fit_record(pipe, train[features], "train", y=train[targets])
mlca_runtime.save_model(model, "model.pkl")
print(f"train={train.shape} features={len(features)} targets={len(targets)}")

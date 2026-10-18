import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
missing = [k for k in ("targets", "excluded_columns", "task") if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
train = pd.read_parquet(inp["train"])


def is_key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.notna().sum() > 1 and s.nunique(dropna=True) == s.notna().sum()
    if pd.api.types.is_integer_dtype(s):
        return len(s) > 1 and s.nunique(dropna=True) == s.notna().sum()
    return False


drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in drop and not is_key_like(train[c])]
bad_features = (set(feature_cols) & set(targets)) | (set(feature_cols) & set(excluded))
if bad_features:
    raise ValueError(f"Target/excluded columns reached features: {sorted(bad_features)}")
X = train[feature_cols]
y = train[targets]

pre = pickle.loads(inp["05_preprocess_fit_preprocessing_X2VH4MDF/preprocessor.pkl"].read_bytes())
sel = pickle.loads(inp["06_features_retain_features_B66XBBY7/selector_all.pkl"].read_bytes())


def fnames(obj):
    return list(obj.feature_names_in_) if hasattr(obj, "feature_names_in_") else None


def includes(obj, other):
    if not isinstance(obj, Pipeline):
        return False
    for step in obj.named_steps.values():
        if step is other:
            return True
        if type(step) is type(other):
            a, b = fnames(step), fnames(other)
            if a is not None and b is not None and a == b:
                return True
    return False


pre_names, sel_names = fnames(pre), fnames(sel)
for nm in (pre_names, sel_names):
    if nm is not None:
        if not set(nm).issubset(set(train.columns)):
            raise ValueError("Saved object expects unknown columns")
        if set(nm) & drop:
            raise ValueError("Saved object expects a target/excluded/group column")

if isinstance(sel, Pipeline) and includes(sel, pre):
    frontend = sel
elif sel_names == feature_cols and isinstance(sel, Pipeline) and len(sel.named_steps) > 1:
    frontend = sel
elif pre_names == feature_cols:
    frontend = pre
elif sel_names is not None and set(sel_names) == set(feature_cols):
    frontend = sel
else:
    raise ValueError("No saved frontend accepts the raw feature contract")

fn = fnames(frontend)
if fn is not None and list(fn) != feature_cols:
    raise ValueError("Frontend feature order does not match contract")

steps = [("frontend", frontend)]
if frontend is pre and sel is not None and not includes(sel, pre):
    if sel_names is None or set(sel_names) != set(feature_cols):
        steps.append(("selector", sel))
steps.append(("model", LinearRegression()))
pipeline = Pipeline(steps)

try:
    probe = frontend.transform(X.head(min(5, len(X))))
    shape = getattr(probe, "shape", None)
    if shape is not None and len(shape) > 1 and shape[1] <= 300:
        frontend.set_output(transform="pandas")
except Exception:
    pass

mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(pipeline, "model.pkl")
print("train_shape", train.shape, "features", len(feature_cols), "targets", targets)
print("saved model.pkl")

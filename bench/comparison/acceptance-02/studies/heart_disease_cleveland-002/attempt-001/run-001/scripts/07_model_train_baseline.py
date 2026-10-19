import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
missing = [k for k in ("targets", "excluded_columns", "task") if k not in proto]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available keys: {list(proto.keys())}")
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
task = proto["task"]
group_col = proto.get("group_column")

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_5Z83A35F/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_Z9TAK0X7/selector_all.pkl")
train = pd.read_parquet(inputs["train"])

def input_features(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        try:
            return list(obj.feature_names_in_)
        except Exception:
            pass
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            f = input_features(step)
            if f is not None:
                return f
    return None

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feat = input_features(pre) or input_features(sel)
if feat is None:
    feat = []
    n = len(train)
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        nun = s.nunique(dropna=False)
        key_like = False
        if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
            key_like = nun == n
        elif pd.api.types.is_integer_dtype(s.dtype):
            vals = s.dropna().unique()
            if len(vals) == n and not set(vals).issubset({0, 1}):
                sequential = vals.max() - vals.min() + 1 == n
                name_id = any(t in c.lower() for t in ("id", "code", "key"))
                key_like = sequential or name_id
        if not key_like:
            feat.append(c)
feat = [c for c in feat if c in train.columns and c not in drop]
if not feat:
    raise ValueError("No feature columns available")
X = train.loc[:, feat].copy()
y = train.loc[:, targets].copy()

steps = []
if pre is not None:
    steps.append(("preprocessor", pre))
    if isinstance(pre, Pipeline) and "selector" in pre.named_steps:
        sel = None
    elif sel is not None:
        sel_in = input_features(sel)
        pre_in = input_features(pre)
        if sel_in is not None and pre_in is not None and set(sel_in) == set(pre_in):
            sel = None
        else:
            if hasattr(pre, "get_feature_names_out") and sel_in is not None:
                try:
                    out_names = list(pre.get_feature_names_out())
                    if set(sel_in) != set(out_names):
                        sel = None
                except Exception:
                    pass
            if sel is not None and hasattr(sel, "transform"):
                steps.append(("selector", sel))
else:
    num_cols = [c for c in feat if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_datetime64_any_dtype(X[c])]
    cat_cols = [c for c in feat if c not in num_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    ct = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
    ct.set_output(transform="pandas")
    steps.append(("preprocessor", ct))

clf = MultiOutputClassifier(LogisticRegression(max_iter=2000))
steps.append(("model", clf))
pipe = Pipeline(steps)
mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(pipe, "model.pkl")
print(f"train={len(train)} features={len(feat)} targets={len(targets)}")

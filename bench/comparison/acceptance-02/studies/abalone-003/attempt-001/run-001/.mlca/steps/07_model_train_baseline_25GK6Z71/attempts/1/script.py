import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"missing protocol key {key}; available: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

drop = set(targets) | set(excluded) | ({group} if group is not None else set())
features = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.is_unique and (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)):
        continue
    features.append(c)
if not features:
    raise ValueError("no feature columns after exclusions")

X_train, y_train = train[features].copy(), train[targets].copy()
X_val, y_val = val[features].copy(), val[targets].copy()

def load_pkl(key):
    p = paths.get(key)
    return pickle.loads(p.read_bytes()) if p is not None else None

def in_names(obj):
    try:
        return list(obj.feature_names_in_)
    except Exception:
        return None

def has_pre(obj):
    if not isinstance(obj, Pipeline):
        return False
    return any(n.lower().startswith("pre") or isinstance(s, ColumnTransformer) for n, s in obj.named_steps.items())

pre = load_pkl("05_preprocess_fit_preprocessing_GTAJ2HZB/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_XC2ENAE2/selector_all.pkl")
for obj in (pre, sel):
    if obj is not None and hasattr(obj, "set_output"):
        try:
            obj.set_output(transform="pandas")
        except Exception:
            pass
    n = in_names(obj) if obj is not None else None
    if n and set(n) & set(targets):
        raise ValueError("a saved transformer expects a target column")

pre_names, sel_names = in_names(pre), in_names(sel)
pre_ok = pre is not None and pre_names is not None and set(pre_names).issubset(set(features))
sel_raw = sel is not None and ((sel_names is not None and set(sel_names).issubset(set(features))) or (sel_names is None and has_pre(sel)))

if sel_raw:
    model = Pipeline([("selector", sel), ("model", LinearRegression())])
elif sel is not None and pre_ok:
    model = Pipeline([("preprocessor", pre), ("selector", sel), ("model", LinearRegression())])
elif pre_ok:
    model = Pipeline([("preprocessor", pre), ("model", LinearRegression())])
else:
    num = [c for c in features if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c])]
    cat = [c for c in features if c not in num]
    max_cat = max(5, int(0.05 * len(train)))
    cat = [c for c in cat if train[c].nunique(dropna=False) <= max_cat]
    parts = []
    if num:
        parts.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    ct = ColumnTransformer(parts, remainder="drop")
    try:
        ct.set_output(transform="pandas")
    except Exception:
        pass
    model = Pipeline([("preprocessor", ct), ("model", LinearRegression())])

try:
    model.set_output(transform="pandas")
except Exception:
    pass

fitted = mlca_runtime.fit_record(model, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "model.pkl")

pred = np.asarray(fitted.predict(X_val))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
yv = y_val[targets].to_numpy()
for j, t in enumerate(targets):
    mlca_runtime.record_metric(f"val_rmse_{t}", float(np.sqrt(np.mean((pred[:, j] - yv[:, j]) ** 2))), "val")
headline = float(np.sqrt(np.mean((pred - yv) ** 2)))
mlca_runtime.record_metric("val_rmse", headline, "val")
print(f"train_rows={len(train)} features={len(features)} targets={targets}")
print(f"val_rmse={headline:.6f}")

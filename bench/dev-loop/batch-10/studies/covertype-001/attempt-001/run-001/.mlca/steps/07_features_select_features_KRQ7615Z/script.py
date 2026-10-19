import mlca_runtime
import json
import pickle
from functools import partial
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

protocol = json.loads(paths["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns") or [])
group = protocol.get("group_column")

train = pd.read_parquet(paths["train"])

with open(paths["06_preprocess_fit_preprocessing_XZ1WZNJH/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

drop = set(targets) | set(excluded)
if group:
    drop.add(group)

feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    is_text = pd.api.types.is_string_dtype(s.dtype) or s.dtype == object
    if is_text and s.nunique(dropna=False) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) > 2:
        if s.nunique(dropna=False) == len(s) and s.min() >= 0 and s.is_monotonic_increasing:
            continue
    feature_cols.append(c)

if hasattr(preprocessor, "feature_names_in_"):
    expected = list(preprocessor.feature_names_in_)
    missing = [c for c in expected if c not in train.columns]
    if missing:
        raise RuntimeError(f"Preprocessor expects missing columns: {missing}")
    feature_cols = expected

X_train = train[feature_cols]
y_train = train[targets[0]].to_numpy()

try:
    t = preprocessor.transform(X_train.iloc[:5])
    n_out = t.shape[1] if hasattr(t, "shape") else len(t[0])
except Exception:
    n_out = X_train.shape[1]

k = min(20, int(n_out))

score_func = partial(mutual_info_classif, random_state=42, discrete_features=False)
selector = SelectKBest(score_func=score_func, k=k)
pipe = Pipeline([("pre", preprocessor), ("select", selector)])

mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(pipe, "selector.pkl")

n_sel = int(pipe.named_steps["select"].get_support().sum())
mlca_runtime.record_metric("n_selected_features", float(n_sel), "train")
print(f"feat_in={len(feature_cols)} enc_out={n_out} k={k} selected={n_sel} rows={len(train)}")

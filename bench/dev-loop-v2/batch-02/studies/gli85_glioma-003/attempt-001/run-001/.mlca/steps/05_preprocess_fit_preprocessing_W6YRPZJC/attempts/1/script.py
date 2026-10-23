import mlca_runtime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
protocol = json.loads(Path(inputs['_protocol']).read_text())

for key in ('targets', 'excluded_columns', 'task'):
    if key not in protocol:
        raise KeyError(f"Missing {key}. Available keys: {list(protocol.keys())}")
targets = protocol['targets']
excluded_cols = list(protocol['excluded_columns'])
group_col = protocol.get('group_column')
outcome = targets[0]
y_train = train[outcome]
is_binary = pd.api.types.is_numeric_dtype(y_train) and y_train.dropna().isin([0, 1]).all()
feature_cols = [c for c in train.columns if c not in targets and c not in excluded_cols and c != group_col]

def is_key_like(s):
    try:
        if s.dtype == object or str(s.dtype) == 'string':
            return s.dropna().nunique() == len(s) and s.notna().all()
        if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
            return s.nunique(dropna=False) == len(s)
    except Exception:
        pass
    return False

key_like = [c for c in feature_cols if is_key_like(train[c])]
feature_cols = [c for c in feature_cols if c not in key_like]
if key_like:
    print(f"key-like excluded: {key_like}")

derived = []
if is_binary:
    yv = y_train.astype(float).values
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            if np.array_equal(train[c].astype(float).values, yv):
                derived.append(c)
    zc = [c for c in feature_cols
          if pd.api.types.is_numeric_dtype(train[c]) and train[c].notna().all()
          and (train[c].astype(float).values >= 0).all()
          and (train[c].astype(float).values[yv == 0] == 0).all()]
    if 0 < len(zc) <= 200:
        data = {c: train[c].astype(float).values for c in zc}
        for i, a in enumerate(zc):
            hit = False
            for b in zc[i+1:]:
                if np.array_equal(data[a] + data[b], yv):
                    derived.extend([a, b]); hit = True; break
            if hit:
                break
derived = sorted(set(derived))
feature_cols = [c for c in feature_cols if c not in derived]
if derived:
    print(f"derived excluded: {derived}")

if not is_binary and pd.api.types.is_numeric_dtype(y_train) and y_train.nunique() > 2:
    yv = y_train.astype(float).values
    warn = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])
            and abs(np.corrcoef(train[c].astype(float).values, yv)[0, 1]) >= 0.98]
    if warn:
        print(f"corr>=0.98 kept: {warn[:10]}")

num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and train[c].notna().any()]
cat_cols = [c for c in feature_cols if c not in num_cols]
datetime_cols = [c for c in cat_cols if pd.api.types.is_datetime64_any_dtype(train[c])]
if datetime_cols:
    raise RuntimeError(f"datetime not supported portably: {datetime_cols}")
thr = max(5, int(0.05 * len(train)))
low_card = [c for c in cat_cols if train[c].nunique(dropna=True) <= thr]
use_cols = num_cols + low_card
if not use_cols:
    raise RuntimeError("no features left")

transformers = []
if num_cols:
    transformers.append(('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                                          ('sc', StandardScaler())]), num_cols))
if low_card:
    transformers.append(('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                                          ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), low_card))
pre = ColumnTransformer(transformers, remainder='drop')
X_train = train[use_cols]
mlca_runtime.fit_record(pre, X_train, 'train')

try:
    n_out = len(pre.get_feature_names_out())
except Exception:
    n_out = 0
if 0 < n_out <= 300:
    try:
        pre.set_output(transform='pandas')
        _ = pre.transform(X_train.iloc[:3])
    except Exception:
        pass

X_val = val.reindex(columns=use_cols, fill_value=np.nan)
try:
    _ = pre.transform(X_val)
    val_ok = True
except Exception as e:
    val_ok = False
    print(f"val error: {e}")

mlca_runtime.save_model(pre, "preprocessor.pkl")
mlca_runtime.record_metric("n_features_in", len(use_cols), "train")
mlca_runtime.record_metric("n_features_out", int(n_out), "train")
print(f"preprocessor saved: in {len(use_cols)} -> out {n_out}, val_ok={val_ok}, train={X_train.shape}")

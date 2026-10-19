import itertools
import json
import numpy as np
import pandas as pd
import mlca_runtime

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
if "targets" not in protocol:
    raise ValueError(f"Missing targets; available keys: {list(protocol)}")
targets = protocol["targets"]
target = targets[0]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
raw_drop = set(targets) | set(excluded)
if group:
    raw_drop.add(group)

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
candidate = [c for c in train.columns if c not in raw_drop]

date_cols = [c for c in candidate if pd.api.types.is_datetime64_any_dtype(train[c])]
for c in date_cols:
    for frame in (train, val):
        if c in frame.columns:
            dt = pd.to_datetime(frame[c])
            frame[c + "_days"] = (dt - pd.Timestamp("1970-01-01")).dt.days
            frame[c + "_month"] = dt.dt.month
            frame[c + "_dow"] = dt.dt.dayofweek
    candidate.remove(c)
    candidate.extend([c + "_days", c + "_month", c + "_dow"])

def key_like(name, s):
    if s.nunique(dropna=False) != len(s) or len(s) <= 1:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s):
        u = np.sort(s.dropna().unique())
        return len(u) == len(s) and np.all(np.diff(u) == 1)
    return False

def is_cat(s):
    return (not pd.api.types.is_numeric_dtype(s)) or pd.api.types.is_bool_dtype(s)

feature_cols = []
high_card = []
for c in candidate:
    s = train[c]
    if key_like(c, s):
        continue
    if is_cat(s) and s.nunique(dropna=True) > 0.05 * len(train):
        high_card.append(c)
        continue
    feature_cols.append(c)

X_screen = train[feature_cols].copy()
y = train[target]
deriv = {}
for c in feature_cols:
    try:
        if (X_screen[c].reset_index(drop=True).to_numpy() == y.reset_index(drop=True).to_numpy()).all():
            deriv[c] = "equals target"
    except Exception:
        pass

num_nonneg = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and (train[c].fillna(0) >= 0).all()]
for k in (2, 3):
    for comb in itertools.combinations(num_nonneg, k):
        if (train[list(comb)].sum(axis=1) == y).all():
            for c in comb:
                deriv[c] = "sum to target"
            break

feature_cols = [c for c in feature_cols if c not in deriv]
X_tr = train[feature_cols].copy()
X_va = val[feature_cols].copy()

tokens = {"?", "-", "NA", "N/A", ""}
found = set()
for c in feature_cols:
    if not is_cat(X_tr[c]):
        continue
    for frame in (X_tr, X_va):
        s = frame[c].astype("string").str.strip()
        mask = s.isin(tokens)
        if mask.any():
            found.add(c)
        frame[c] = s.mask(mask | s.isna(), np.nan).astype(object)

num_cols = [c for c in feature_cols if not is_cat(X_tr[c])]
cat_cols = [c for c in feature_cols if is_cat(X_tr[c])]
width = len(num_cols) + sum(int(X_tr[c].nunique(dropna=True)) for c in cat_cols)

transformers = []
if num_cols:
    transformers.append(("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat_cols))
if not transformers:
    raise ValueError("No usable feature columns after screening")

preprocessor = ColumnTransformer(
    transformers, remainder="drop", verbose_feature_names_out=False, sparse_threshold=0.0
)
if width <= 300:
    preprocessor.set_output(transform="pandas")

preprocessor = mlca_runtime.fit_record(preprocessor, X_tr, "train")
X_va_t = preprocessor.transform(X_va)
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

print(f"placeholders: {len(found)} columns {sorted(found)}")
print(f"excluded derivations: {deriv if deriv else 'none'}")
print(f"features: {len(feature_cols)} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")
print(f"high-card dropped: {high_card if high_card else 'none'}")
print(f"val transformed: {X_va_t.shape}")

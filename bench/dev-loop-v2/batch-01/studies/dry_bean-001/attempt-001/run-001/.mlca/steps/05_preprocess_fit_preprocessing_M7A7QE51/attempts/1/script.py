import json
import itertools
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

def need(key):
    if key not in proto:
        raise KeyError(f"Missing {key}; available top-level keys: {sorted(proto.keys())}")
    return proto[key]

targets = need("targets")
excluded = need("excluded_columns")
group = need("group_column")
task = need("task")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

protected = set(targets) | set(excluded)
if group is not None:
    protected.add(group)

def key_like(s):
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        n = str(s.name).lower()
        return s.nunique(dropna=True) == len(s) and any(t in n for t in ("id", "key", "code", "index", "seq"))
    return False

feature_cols = [c for c in train.columns if c not in protected and not key_like(train[c])]

ph = {"?", "-", "NA", "N/A"}
found = {}
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        t = s.dropna().astype(str).str.strip()
        m = t.isin(ph) | t.eq("")
        if m.any():
            found[c] = int(m.sum())
print("Placeholder tokens found in %d columns%s" % (
    len(found), (": " + ", ".join(f"{k}({v})" for k, v in found.items())) if found else ""))

excluded_features = []
if len(targets) == 1:
    y = train[targets[0]]
    for c in feature_cols:
        if train[c].astype(object).eq(y.astype(object)).fillna(False).all():
            excluded_features.append((c, "exact target copy"))
    if pd.api.types.is_numeric_dtype(y):
        cand = [c for c in feature_cols
                if c not in {e[0] for e in excluded_features}
                and pd.api.types.is_numeric_dtype(train[c])
                and train[c].ge(0).all()]
        deriv = set()
        for r in (1, 2, 3):
            for combo in itertools.combinations(cand, r):
                if train[list(combo)].sum(axis=1).eq(y).fillna(False).all():
                    deriv.update(combo)
                    break
            if deriv:
                break
        for c in deriv:
            excluded_features.append((c, "target is row sum of non-negative columns"))

excluded_set = {c for c, _ in excluded_features}
feature_cols = [c for c in feature_cols if c not in excluded_set]
if excluded_features:
    print("Excluded: " + "; ".join(f"{c} ({r})" for c, r in excluded_features))
else:
    print("No target-derived features excluded.")

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()

date_cols = [c for c in feature_cols if pd.api.types.is_datetime64_any_dtype(train[c])]
for c in date_cols:
    X_train[c + "_month"] = train[c].dt.month
    X_val[c + "_month"] = val[c].dt.month
feature_cols = [c for c in feature_cols if c not in date_cols] + [c + "_month" for c in date_cols]
X_train = X_train[feature_cols]
X_val = X_val[feature_cols]

num_cols, cat_cols, drop_cols = [], [], []
for c in feature_cols:
    s = train[c] if c in train.columns else X_train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif pd.api.types.is_string_dtype(s) or s.dtype == object:
        if s.nunique(dropna=True) > 0.05 * len(train):
            drop_cols.append(c)
        else:
            cat_cols.append(c)
    else:
        drop_cols.append(c)

if drop_cols:
    feature_cols = [c for c in feature_cols if c not in drop_cols]
    X_train = X_train[feature_cols]
    X_val = X_val[feature_cols]

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))

preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
width = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
if width <= 300:
    preprocessor.set_output(transform="pandas")
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")

print(f"Features: {len(feature_cols)} | numeric={len(num_cols)} categorical={len(cat_cols)}")
y_train = train[targets[0]] if len(targets) == 1 else train[targets].copy()
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train", y_train)
X_val_t = preprocessor.transform(X_val)
print(f"Val transformed shape: {X_val_t.shape}")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

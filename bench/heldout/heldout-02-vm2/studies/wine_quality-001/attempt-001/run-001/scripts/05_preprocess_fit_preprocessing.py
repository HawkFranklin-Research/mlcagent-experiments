import mlca_runtime
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
proto = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in proto:
        raise KeyError(f"Missing {k}; available top-level keys: {list(proto)}")
targets = proto["targets"]
excluded = list(proto["excluded_columns"])
group = proto.get("group_column")

def is_text(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

protected = set(targets) | set(excluded)
base = [c for c in train.columns if c not in protected and c != group]

def key_like(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return False
    if not (is_text(s) or pd.api.types.is_integer_dtype(s)):
        return False
    n = s.notna().sum()
    return n > 0 and s.nunique(dropna=True) == n
key_drop = [c for c in base if key_like(train[c])]
base = [c for c in base if c not in key_drop]

high = [c for c in base if (is_text(train[c]) or isinstance(train[c].dtype, pd.CategoricalDtype))
        and train[c].nunique(dropna=True) > 0.05 * len(train)]
base = [c for c in base if c not in high]

derived = []
for t in targets:
    y = train[t]
    num_ok = pd.api.types.is_numeric_dtype(y)
    derived += [c for c in base if c not in derived and train[c].eq(y).all()]
    cand = [c for c in base if c not in derived]
    if num_ok:
        nonneg = [c for c in cand if pd.api.types.is_numeric_dtype(train[c]) and train[c].min(skipna=True) >= 0]
        found = True
        while found:
            found = False
            for r in (2, 3):
                for combo in itertools.combinations(nonneg, r):
                    if set(combo) & set(derived):
                        continue
                    if train[list(combo)].sum(axis=1).eq(y).all():
                        derived += list(combo)
                        found = True
                        break
                if found:
                    break
            nonneg = [c for c in nonneg if c not in derived]
base = [c for c in base if c not in derived]

for t in targets:
    y = train[t]
    if pd.api.types.is_numeric_dtype(y):
        warn = [c for c in base if pd.api.types.is_numeric_dtype(train[c]) and abs(train[c].corr(y)) >= 0.98]
        if warn:
            print("Corr>=0.98 retained: " + ", ".join(map(str, warn)))

feature_cols = base
num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

ph_tokens = ["?", "-", "NA", "N/A", ""]
ph_cols = []
for c in cat_cols:
    raw = train[c].dropna().astype(str)
    if (raw != raw.str.strip()).any() or raw.str.strip().isin(ph_tokens).any():
        ph_cols.append(c)
if ph_cols:
    print(f"Placeholder/whitespace columns: {len(ph_cols)}")

n_out = len(num_cols) + sum(max(1, train[c].nunique(dropna=True)) for c in cat_cols)
wide = n_out > 300

num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_steps = []
if ph_cols:
    cat_steps.append(("strip", FunctionTransformer(
        func=pd.DataFrame.replace, kw_args={"to_replace": r"^\s+|\s+$", "value": "", "regex": True})))
    cat_steps.append(("miss", FunctionTransformer(
        func=pd.DataFrame.replace, kw_args={"to_replace": ph_tokens, "value": np.nan, "regex": False})))
cat_steps.append(("impute", SimpleImputer(strategy="most_frequent")))
cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=wide)))
cat_pipe = Pipeline(cat_steps)

transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
ct = ColumnTransformer(transformers, remainder="drop")
if not wide:
    ct.set_output(transform="pandas")

X_train = train[feature_cols]
X_val = val[feature_cols]
mlca_runtime.fit_record(ct, X_train, "train")
Xt = ct.transform(X_val)
mlca_runtime.save_model(ct, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_transformed_cols", float(Xt.shape[1]), "val")

print(f"Features: {len(feature_cols)} num={len(num_cols)} cat={len(cat_cols)}")
print(f"Dropped key/high/derived: {len(key_drop)}/{len(high)}/{len(derived)}")
if derived:
    print("Derived from target (excluded): " + ", ".join(map(str, derived)))
print(f"Val transformed: {Xt.shape}")

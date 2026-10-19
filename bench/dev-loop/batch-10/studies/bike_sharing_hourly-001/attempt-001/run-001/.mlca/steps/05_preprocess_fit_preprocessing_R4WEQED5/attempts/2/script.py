import mlca_runtime
import json, functools
from itertools import combinations
from operator import attrgetter
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
train = pd.read_parquet(paths["train"])

for k in ("targets", "excluded_columns", "task"):
    if k not in protocol:
        raise KeyError(f"missing {k}; available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
target = targets[0]

skip = set(targets) | set(excluded) | ({group} if group else set())
features = [c for c in train.columns if c not in skip]

keep = []
for c in features:
    s = train[c]
    if s.dtype == object and s.is_unique:
        print(f"drop {c}: unique text"); continue
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.is_unique:
        print(f"drop {c}: unique int"); continue
    keep.append(c)
features = keep

date_cols = [c for c in features
             if c == "dteday" or pd.api.types.is_datetime64_any_dtype(train[c])]
other = [c for c in features if c not in date_cols]

pos_cols = [c for c in other if pd.api.types.is_numeric_dtype(train[c]) and (train[c] >= 0).all()]
derived = set()
for k in (1, 2, 3):
    for combo in combinations(pos_cols, k):
        if (train[list(combo)].sum(axis=1) == train[target]).all():
            derived.update(combo)
for c in sorted(derived):
    print(f"drop {c}: derivation of {target}")
other = [c for c in other if c not in derived]

if pd.api.types.is_numeric_dtype(train[target]):
    for c in other:
        if pd.api.types.is_numeric_dtype(train[c]):
            r = train[c].corr(train[target])
            if pd.notna(r) and abs(r) >= 0.98:
                print(f"high corr {c}: {r:.3f}")

phs = ["?", "-", "NA", "N/A"]
text_cols = [c for c in features if train[c].dtype == object]
ph_cols = []
for c in text_cols:
    for ph in phs:
        if (train[c] == ph).any():
            ph_cols.append(c); break

n = len(train)
num_cols, cat_cols, hi_card = [], [], []
for c in other:
    if pd.api.types.is_numeric_dtype(train[c]):
        num_cols.append(c)
    elif train[c].nunique() > 0.05 * n:
        hi_card.append(c)
    else:
        cat_cols.append(c)
for c in hi_card:
    print(f"drop {c}: high-cardinality text")

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

def date_pipe(attr):
    return Pipeline([
        ("ravel", FunctionTransformer(np.ravel, validate=False)),
        ("parse", FunctionTransformer(pd.to_datetime, kw_args={"errors": "coerce"}, validate=False)),
        ("extract", FunctionTransformer(attrgetter(attr), validate=False)),
        ("to_num", FunctionTransformer(pd.to_numeric, kw_args={"errors": "coerce"}, validate=False)),
        ("reshape", FunctionTransformer(np.reshape, kw_args={"newshape": (-1, 1)}, validate=False)),
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])

transformers = []
used = []
for c in date_cols:
    transformers.append((f"date_{c}_m", date_pipe("month"), [c]))
    transformers.append((f"date_{c}_dow", date_pipe("dayofweek"), [c]))
    used.append(c)
if num_cols:
    transformers.append(("num", num_pipe, num_cols)); used += num_cols
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols)); used += cat_cols

ct = ColumnTransformer(transformers, remainder="drop")
ct.set_output(transform="pandas")

if ph_cols:
    phs_map = {p: np.nan for p in phs}
    cleaner = FunctionTransformer(
        functools.partial(pd.DataFrame.replace, to_replace=phs_map),
        validate=False,
    )
    pipe = Pipeline([("clean", cleaner), ("ct", ct)])
else:
    pipe = Pipeline([("ct", ct)])
pipe.set_output(transform="pandas")

X_train = train[used]
mlca_runtime.fit_record(pipe, X_train, partition="train")
mlca_runtime.save_model(pipe, "preprocessor.pkl")

out = pipe.transform(X_train.head(2))
print(f"features: {len(used)}, out cols: {out.shape[1]}, placeholders in {len(ph_cols)} cols")

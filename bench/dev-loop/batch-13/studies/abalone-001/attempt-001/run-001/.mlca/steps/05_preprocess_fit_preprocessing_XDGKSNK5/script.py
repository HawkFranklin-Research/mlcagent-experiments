import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
protocol = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in protocol:
        raise KeyError(f"Missing '{k}'; available top-level keys: {sorted(protocol)}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"] or [])
group = protocol["group_column"]


def is_text(s):
    return s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)


def key_like(s):
    n = int(s.notna().sum())
    if n < 2:
        return False
    if is_text(s):
        return s.nunique(dropna=True) == n
    if pd.api.types.is_integer_dtype(s):
        v = s.dropna()
        if v.nunique() == len(v):
            name = str(s.name).lower()
            if any(t in name for t in ("id", "key", "code", "index")):
                return True
            d = v.diff().dropna()
            if len(d) and (d == 1).all():
                return True
    return False


drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
features = [c for c in train.columns if c not in drop and not key_like(train[c])]

num_all = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]


def same(a, b):
    a = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    return len(a) == len(b) and np.array_equal(a, b, equal_nan=True)


derived = {}
for t in targets:
    if t not in train:
        continue
    y = train[t]
    for c in list(features):
        if train[c].equals(y) or (c in num_all and pd.api.types.is_numeric_dtype(y) and same(train[c], y)):
            derived[c] = f"equals {t}"
    if pd.api.types.is_numeric_dtype(y):
        for r in (2, 3):
            for combo in combinations(num_all, r):
                if any(c in derived for c in combo):
                    continue
                if all(train[c].notna().all() and (train[c] >= 0).all() for c in combo):
                    if same(train[list(combo)].sum(axis=1), y):
                        for c in combo:
                            derived.setdefault(c, f"part of sum equal to {t}")

if derived:
    print("Excluded target derivations:", {c: derived[c] for c in derived})
features = [c for c in features if c not in derived]

warn = []
for t in targets:
    if t not in train or not pd.api.types.is_numeric_dtype(train[t]):
        continue
    for c in features:
        if c in num_all:
            r = train[c].corr(train[t])
            if pd.notna(r) and abs(r) >= 0.98:
                warn.append(f"{c}({r:.2f})")
if warn:
    print("High correlation, kept:", ", ".join(sorted(set(warn))))

dt_cols = [c for c in features if pd.api.types.is_datetime64_any_dtype(train[c])]
if dt_cols:
    features = [c for c in features if c not in dt_cols]
    print("Dropped datetime columns:", dt_cols)

n = len(train)
num_cols, cat_cols, dropped = [], [], []
for c in features:
    s = train[c]
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif is_text(s):
        (cat_cols if s.nunique(dropna=True) <= 0.05 * n else dropped).append(c)
    else:
        dropped.append(c)
if dropped:
    print("Dropped high-cardinality/unsupported columns:", dropped)

tokens = ["?", "-", "NA", "N/A"]
present, hits, has_ws = [], {}, False
for c in cat_cols:
    s = train[c].astype("string")
    raw = s.dropna()
    found = [t for t in tokens if bool((raw == t).any())]
    present.extend(found)
    if found or bool((raw.str.strip() == "").any()):
        hits[c] = int(raw.isin(tokens + [""]).sum() + (raw == "").sum())
    if bool((raw != raw.str.strip()).any()):
        has_ws = True
present = sorted(set(present))
if hits or has_ws:
    print(f"Placeholder tokens in {len(hits)} columns:", sorted(hits))

steps = [("fill", SimpleImputer(strategy="constant", fill_value="__missing__"))]
if has_ws:
    steps.append(("strip", FunctionTransformer(np.char.strip, feature_names_out="one-to-one")))
for i, t in enumerate(present + [""]):
    steps.append((f"tok{i}", SimpleImputer(missing_values=t, strategy="constant", fill_value="__missing__")))
steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))

transformers = []
if num_cols:
    transformers.append(("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline(steps), cat_cols))

pre = ColumnTransformer(transformers, remainder="drop")
pre.set_output(transform="pandas")
model_features = num_cols + cat_cols
X_train = train[model_features]
mlca_runtime.fit_record(pre, X_train, "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
print(f"Saved preprocessor.pkl rows={len(X_train)} numeric={len(num_cols)} categorical={len(cat_cols)}")

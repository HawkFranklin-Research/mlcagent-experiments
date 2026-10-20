import mlca_runtime
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing _protocol['{k}']; available top-level keys: {sorted(d.keys())}")
    return d[k]

if "targets" in protocol:
    targets = req(protocol, "targets")
elif isinstance(protocol.get("task"), dict) and "targets" in protocol["task"]:
    targets = protocol["task"]["targets"]
else:
    raise KeyError(f"Missing _protocol['targets']; available top-level keys: {sorted(protocol.keys())}")
excluded = req(protocol, "excluded_columns")
group = protocol.get("group_column")

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
candidates = [c for c in train.columns if c not in drop]
n = len(train)
feature_cols = []
for c in candidates:
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == n:
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == n and ("id" in c.lower() or "code" in c.lower()):
        continue
    feature_cols.append(c)

derived = {}
for t in targets:
    if t not in train:
        continue
    y = train[t]
    for c in feature_cols:
        s = train[c]
        if pd.api.types.is_numeric_dtype(s) and not s.isna().any() and (s == y).all():
            derived[c] = f"equals {t}"
    nonneg = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and not train[c].isna().any() and (train[c] >= 0).all()]
    for r in (2, 3):
        for combo in combinations(nonneg, r):
            if (train[list(combo)].sum(axis=1) == y).all():
                for c in combo:
                    derived.setdefault(c, f"sum{combo} equals {t}")
if derived:
    feature_cols = [c for c in feature_cols if c not in derived]
    shown = "; ".join(f"{c} ({derived[c]})" for c in list(derived)[:5])
    print(f"excluded derivations: {shown}" + (" ..." if len(derived) > 5 else ""))

placeholders = ["?", "-", "NA", "N/A"]
ph_cols = set()
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype("string").str.strip()
        if any((stripped == ph).any() for ph in placeholders) or stripped.str.fullmatch(r"\s*").any():
            ph_cols.add(c)
if ph_cols:
    print(f"placeholder tokens in {len(ph_cols)} columns")

numeric_cols, cat_cols, dropped = [], [], []
thr = max(5, 0.05 * len(train))
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dropped.append(c)
    elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    elif s.nunique(dropna=True) > thr:
        dropped.append(c)
    else:
        cat_cols.append(c)
feature_cols = numeric_cols + cat_cols
if dropped:
    print(f"dropped datetime/high-cardinality columns: {len(dropped)}")

X_train = train[feature_cols].copy()
width = len(numeric_cols) + sum(X_train[c].nunique(dropna=True) for c in cat_cols)
pandas_out = width <= 300
transformers = []
if numeric_cols:
    transformers.append(("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_cols))
if cat_cols:
    cat_steps = []
    for i, ph in enumerate(["?", "-", "NA", "N/A", ""]):
        cat_steps.append((f"ph_{i}", SimpleImputer(missing_values=ph, strategy="constant", fill_value=np.nan)))
    cat_steps.extend([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not pandas_out)),
    ])
    transformers.append(("cat", Pipeline(cat_steps), cat_cols))

ct = ColumnTransformer(transformers, remainder="drop")
if pandas_out:
    try:
        ct.set_output(transform="pandas")
    except Exception:
        pass

pre = mlca_runtime.fit_record(ct, X_train, "train")
try:
    encoded_width = len(pre.get_feature_names_out())
except Exception:
    encoded_width = width
mlca_runtime.save_model(pre, "preprocessor.pkl")
mlca_runtime.record_metric("n_feature_columns", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_encoded_columns", float(encoded_width), "train")
print(f"saved preprocessor.pkl: {len(feature_cols)} features -> {encoded_width} encoded columns")

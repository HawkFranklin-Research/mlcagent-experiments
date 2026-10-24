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
protocol = json.loads(inp["_protocol"].read_text())

try:
    targets = list(protocol["targets"])
    excluded = list(protocol["excluded_columns"])
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available keys: {list(protocol)}")

group = protocol.get("group_column")
group_cols = [] if group is None else ([group] if isinstance(group, str) else list(group))

def key_like(c):
    s = train[c]
    if s.dtype.kind in "OUS":
        return s.nunique(dropna=True) == len(s) and len(s) > 1
    if pd.api.types.is_integer_dtype(s.dtype):
        name = c.lower()
        return s.nunique(dropna=True) == len(s) and any(k in name for k in ("id", "key", "code"))
    return False

exclude_set = set(targets) | set(excluded) | set(group_cols)
feature_cols = [c for c in train.columns if c not in exclude_set and not key_like(c)]

excluded_deriv = []
if targets:
    for t in targets:
        if t not in train:
            continue
        y = train[t]
        for c in feature_cols:
            if c != t and train[c].equals(y):
                excluded_deriv.append((c, f"equals target {t}"))
        num_candidates = [
            c for c in feature_cols
            if pd.api.types.is_numeric_dtype(train[c]) and (train[c].dropna() >= 0).all()
        ]
        for r in (1, 2, 3):
            for combo in itertools.combinations(num_candidates, r):
                s = train[list(combo)].sum(axis=1, min_count=len(combo))
                if s.equals(y):
                    for c in combo:
                        if c not in {x[0] for x in excluded_deriv}:
                            excluded_deriv.append((c, f"sums to target {t}"))

feat = [c for c in feature_cols if c not in {x[0] for x in excluded_deriv}]

dropped = []
n = len(train)
for c in feat:
    s = train[c]
    if s.dtype.kind in "OUS" and s.nunique(dropna=True) > max(10, 0.05 * n):
        dropped.append(c)
feat = [c for c in feat if c not in dropped]

tokens = ["?", "-", "NA", "N/A"]
ph_cols, found_tokens, ws_cols = set(), set(), set()
for c in feat:
    s = train[c]
    if s.dtype.kind in "OUS":
        ss = s.astype(str)
        stripped = ss.str.strip()
        for tok in tokens:
            if (stripped == tok).any():
                ph_cols.add(c)
                found_tokens.add(tok)
        if (ss != stripped).any():
            ws_cols.add(c)

num_cols, cat_cols = [], []
for c in feat:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        nun = s.nunique(dropna=True)
        if pd.api.types.is_integer_dtype(s.dtype) and nun <= 10 and not set(s.dropna().unique()).issubset({0, 1}):
            cat_cols.append(c)
        else:
            num_cols.append(c)
    else:
        cat_cols.append(c)

log_cols = []
for c in num_cols:
    s = train[c].dropna()
    if len(s) and (s >= 0).all() and s.skew() > 1.0:
        log_cols.append(c)
plain_num = [c for c in num_cols if c not in log_cols]

width = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
low = width <= 300

cat_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
for i, tok in enumerate(sorted(found_tokens)):
    cat_steps.append((f"tok{i}", SimpleImputer(missing_values=tok, strategy="most_frequent")))
if ph_cols or ws_cols:
    cat_steps.append(("strip", FunctionTransformer(np.char.strip, feature_names_out="one-to-one")))
cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not low)))

transformers = []
if log_cols:
    transformers.append((
        "num_log",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]),
        log_cols,
    ))
if plain_num:
    transformers.append((
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        plain_num,
    ))
if cat_cols:
    transformers.append(("cat", Pipeline(cat_steps), cat_cols))

preprocessor = ColumnTransformer(transformers, remainder="drop")
if low:
    preprocessor.set_output(transform="pandas")

preprocessor = mlca_runtime.fit_record(preprocessor, train[feat], "train")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

print(f"features={len(feat)} numeric={len(num_cols)} categorical={len(cat_cols)} log={len(log_cols)}")
print(f"placeholders columns={len(ph_cols | ws_cols)} tokens={sorted(found_tokens)}")
if excluded_deriv:
    print("excluded_derivations=" + ", ".join(f"{c}:{r}" for c, r in excluded_deriv))
if dropped:
    print("dropped_high_card=" + ",".join(dropped))
print(f"encoded_width_est={width} pandas_output={low}")

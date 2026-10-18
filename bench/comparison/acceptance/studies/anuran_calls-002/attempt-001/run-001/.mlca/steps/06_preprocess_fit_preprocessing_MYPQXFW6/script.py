import json, re
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
protocol = json.loads(inputs["_protocol"].read_text())

need = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in need if k not in protocol]
if missing:
    raise ValueError(f"Missing protocol keys {missing}; available: {list(protocol)}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
task = protocol["task"]
task_type = task.get("type", "")
drop = set(targets) | set(excluded) | {group_col}
candidate = [c for c in train.columns if c not in drop]

def is_key_like(col):
    s = train[col]
    if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.nunique(dropna=True) == len(s) and re.search(r"(id|key|code|index|record)", col, re.I) is not None
    return False

candidate = [c for c in candidate if not is_key_like(c)]

tokens = ["?", "-", "NA", "N/A"]
ph = {}
for c in candidate:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s):
        txt, stripped = s.astype(str), s.astype(str).str.strip()
        for tok in tokens:
            if (txt == tok).any() or (stripped == tok).any():
                ph.setdefault(tok, set()).add(c)
        if (stripped == "").any():
            ph.setdefault("whitespace", set()).add(c)
        if (txt != stripped).any():
            ph.setdefault("surrounding whitespace", set()).add(c)
if ph:
    print("Placeholders: " + "; ".join(f"{k} in {len(v)} col(s)" for k, v in ph.items()))

derived = []
for c in candidate:
    for t in targets:
        if train[c].astype(str).equals(train[t].astype(str)):
            derived.append((c, f"exact duplicate of {t}"))
            break
if all(pd.api.types.is_numeric_dtype(train[t]) for t in targets) and task_type not in ("multiclass", "binary", "classification", "categorical"):
    from itertools import combinations
    dnames = {d[0] for d in derived}
    for t in targets:
        if not pd.api.types.is_numeric_dtype(train[t]):
            continue
        nonneg = [c for c in candidate if c not in dnames and pd.api.types.is_numeric_dtype(train[c]) and (train[c] >= 0).all()]
        found = None
        for r in (2, 3):
            for combo in combinations(nonneg, r):
                if (train[list(combo)].sum(axis=1) == train[t]).all():
                    found = (combo[0], f"row sum {combo} equals {t}")
                    break
            if found:
                break
        if found and found[0] not in dnames:
            derived.append(found)
            dnames.add(found[0])
if derived:
    print("Excluded derived: " + "; ".join(f"{n} ({r})" for n, r in derived))

derived_names = {d[0] for d in derived}
feature_cols = [c for c in candidate if c not in derived_names]

if all(pd.api.types.is_numeric_dtype(train[t]) for t in targets) and task_type not in ("multiclass", "binary", "classification", "categorical"):
    for t in targets:
        if pd.api.types.is_numeric_dtype(train[t]):
            numeric_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
            if numeric_feats:
                corr = train[numeric_feats].corrwith(train[t].astype(float)).abs()
                warn = corr[corr >= 0.98].index.tolist()
                if warn:
                    print(f"High correlation with {t} (kept): {', '.join(warn)}")

numeric_cols, categorical_cols, high_card = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    elif s.nunique(dropna=True) > 0.05 * len(train):
        high_card.append(c)
    else:
        categorical_cols.append(c)
if high_card:
    feature_cols = [c for c in feature_cols if c not in high_card]
    print(f"Dropped high-cardinality text: {', '.join(high_card)}")

cat_ok = []
for c in categorical_cols:
    s = train[c]
    if s.notna().any() and (s.astype(str).str.strip() != "").any():
        cat_ok.append(c)
    else:
        feature_cols.remove(c)
categorical_cols = cat_ok

width = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in categorical_cols)
sparse_output = width > 300

num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_steps = [
    ("nan_fill", SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="")),
    ("strip", FunctionTransformer(np.char.strip)),
]
for tok in tokens:
    cat_steps.append((f"replace_{tok}", FunctionTransformer(np.char.replace, kw_args={"old": tok, "new": ""})))
cat_steps += [
    ("imputer", SimpleImputer(missing_values="", strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)),
]
cat_pipeline = Pipeline(cat_steps)

transformers = []
if numeric_cols:
    transformers.append(("num", num_pipeline, numeric_cols))
if categorical_cols:
    transformers.append(("cat", cat_pipeline, categorical_cols))
if not transformers:
    raise ValueError("No feature columns left after screening")

preprocessor = ColumnTransformer(transformers, remainder="drop")
if width <= 300:
    preprocessor.set_output(transform="pandas")

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()
for c in categorical_cols:
    X_train[c] = X_train[c].astype(object)
    X_val[c] = X_val[c].astype(object)

fitted = mlca_runtime.fit_record(preprocessor, X_train, "train")
Xt_val = fitted.transform(X_val)
print(f"Features: {len(feature_cols)} ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical); val transformed {Xt_val.shape}")
mlca_runtime.save_model(fitted, "preprocessor.pkl")

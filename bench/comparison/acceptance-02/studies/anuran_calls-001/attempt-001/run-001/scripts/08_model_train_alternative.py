import json
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

def req(d, key):
    if key not in d:
        raise KeyError(f"Missing {key}; available top-level keys: {sorted(d.keys())}")
    return d[key]

targets = list(req(protocol, "targets"))
excluded = list(req(protocol, "excluded_columns") or [])
group = req(protocol, "group_column")
_ = req(protocol, "task")

train = pd.read_parquet(inp["train"])
missing = [c for c in targets if c not in train.columns]
if missing:
    raise KeyError(f"Missing target columns: {missing}")

def is_date_like(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)):
        return False
    vals = s.dropna()
    if vals.empty:
        return False
    try:
        parsed = pd.to_datetime(vals.head(50), errors="coerce")
        return bool(parsed.notna().all())
    except Exception:
        return False

feature_cols = []
n = len(train)
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    nun = s.nunique(dropna=True)
    if is_date_like(s):
        feature_cols.append(c)
        continue
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)) and nun == n and n > 1:
        continue
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s) and nun == n and n > 1:
        continue
    feature_cols.append(c)

numeric_cols, cat_cols, dropped = [], [], []
limit = max(1, int(0.05 * n))
for c in feature_cols:
    s = train[c]
    if s.notna().sum() == 0 or is_date_like(s):
        dropped.append(c)
    elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        nun = s.nunique(dropna=True)
        if nun == 0 or nun > limit:
            dropped.append(c)
        else:
            cat_cols.append(c)
    else:
        dropped.append(c)

feature_cols = numeric_cols + cat_cols
if not feature_cols:
    raise ValueError("No usable feature columns remain after exclusions")

encoded_width = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
dense_output = encoded_width <= 300
preprocessors = []
if numeric_cols:
    preprocessors.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols))
if cat_cols:
    preprocessors.append((
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output)),
        ]),
        cat_cols,
    ))

ct = ColumnTransformer(preprocessors, remainder="drop", verbose_feature_names_out=False)
if dense_output:
    ct.set_output(transform="pandas")

model = Pipeline([
    ("preprocess", ct),
    ("clf", MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))),
])
if dense_output:
    model.set_output(transform="pandas")

X = train[feature_cols]
y = train[targets].copy()
fitted = mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

print("train rows", len(train), "features", len(feature_cols), "targets", len(targets))
print("encoded width", encoded_width, "dense", dense_output)
print("dropped non-modeled columns", len(dropped))
print("saved alternative_model.pkl")

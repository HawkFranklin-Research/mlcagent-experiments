import json
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

for key in ("targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
train = pd.read_parquet(paths["train"])
n = len(train)

def keylike(s):
    if s.nunique(dropna=False) != n:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s):
        try:
            return (s.max() - s.min() + 1) == n
        except Exception:
            return False
    return False

numeric_cols, cat_cols = [], []
dropped = set(targets) | excluded | ({group} if group else set())

for c in train.columns:
    if c in dropped:
        continue
    s = train[c]
    if keylike(s):
        continue
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    else:
        nunique = s.nunique(dropna=False)
        if nunique <= max(1, int(0.05 * n)):
            cat_cols.append(c)

feature_cols = numeric_cols + cat_cols
if not feature_cols:
    raise ValueError("No usable feature columns after applying protocol exclusions")

X = train[feature_cols].copy()
y = train[targets].copy()

cat_width = sum(train[c].nunique(dropna=False) for c in cat_cols)
width = len(numeric_cols) + cat_width
transformers = []

if numeric_cols:
    transformers.append((
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        numeric_cols,
    ))

if cat_cols:
    transformers.append((
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=width > 300)),
        ]),
        cat_cols,
    ))

pre = ColumnTransformer(transformers, remainder="drop")
if width <= 300 and hasattr(pre, "set_output"):
    pre.set_output(transform="pandas")

model = Pipeline([
    ("preprocessor", pre),
    ("classifier", MultiOutputClassifier(LogisticRegression(max_iter=2000))),
])

fitted = mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(fitted, "model.pkl")

print(f"train rows={len(X)} features={len(feature_cols)} targets={len(targets)}")
print("saved model.pkl")

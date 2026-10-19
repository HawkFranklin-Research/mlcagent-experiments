import json
from pathlib import Path
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(Path(inputs["_protocol"]).read_text())

for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing _protocol[{key!r}]; available keys: {list(protocol)}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
_ = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
y = train[targets]
n = len(train)
candidate = [c for c in train.columns if c not in targets and c not in excluded and c != group]
X = train[candidate].copy()

feature_cols, num_cols, cat_cols = [], [], []
for c in candidate:
    s = X[c]
    if pd.api.types.is_integer_dtype(s) and n > 2 and s.nunique(dropna=True) == n:
        continue
    if pd.api.types.is_datetime64_any_dtype(s):
        parts = pd.DataFrame({
            f"{c}_year": s.dt.year,
            f"{c}_month": s.dt.month,
            f"{c}_day": s.dt.day,
            f"{c}_dow": s.dt.dayofweek,
        }, index=X.index)
        X = pd.concat([X.drop(columns=[c]), parts], axis=1)
        for p in parts.columns:
            feature_cols.append(p)
            num_cols.append(p)
        continue
    low = c.lower()
    if not pd.api.types.is_numeric_dtype(s) and ("date" in low or "time" in low):
        parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().mean() > 0.8:
            parts = pd.DataFrame({
                f"{c}_year": parsed.dt.year,
                f"{c}_month": parsed.dt.month,
                f"{c}_day": parsed.dt.day,
                f"{c}_dow": parsed.dt.dayofweek,
            }, index=X.index)
            X = pd.concat([X.drop(columns=[c]), parts], axis=1)
            for p in parts.columns:
                feature_cols.append(p)
                num_cols.append(p)
            continue
    if pd.api.types.is_numeric_dtype(s):
        feature_cols.append(c)
        num_cols.append(c)
    else:
        nun = s.nunique(dropna=True)
        if nun == 0 or nun == n or nun > max(1, int(0.05 * n)):
            continue
        feature_cols.append(c)
        cat_cols.append(c)

X = X[feature_cols]
width = len(num_cols) + sum(X[c].nunique(dropna=True) for c in cat_cols)
use_pandas = width <= 300

num_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not use_pandas)),
])

transformers = []
if num_cols:
    transformers.append(("num", num_tf, num_cols))
if cat_cols:
    transformers.append(("cat", cat_tf, cat_cols))

pre = ColumnTransformer(
    transformers,
    remainder="drop",
    sparse_threshold=0.3 if use_pandas else 1.0,
)
if use_pandas:
    pre.set_output(transform="pandas")

pipe = Pipeline([
    ("pre", pre),
    ("model", MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))),
])
if use_pandas:
    pipe.set_output(transform="pandas")

model = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(model, "alternative_model.pkl")
print("rows", len(X), "features", len(feature_cols), "num", len(num_cols), "cat", len(cat_cols))
print("saved alternative_model.pkl")

import json
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol["group_column"]
train = pd.read_parquet(inputs["train"])

for target in targets:
    if target not in train.columns:
        raise KeyError(f"Target {target!r} not found in training table")

exclude = set(targets) | set(excluded)
if group_col:
    exclude.add(group_col)

feature_cols = []
n_rows = len(train)
for col in [c for c in train.columns if c not in exclude]:
    s = train[col]
    is_text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)
    if s.nunique(dropna=False) == n_rows and (is_text or pd.api.types.is_integer_dtype(s)):
        continue
    if is_text and s.nunique(dropna=True) > max(5, 0.05 * n_rows):
        continue
    feature_cols.append(col)

def is_categorical(s: pd.Series) -> bool:
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s):
        vals = set(s.dropna().unique())
        if vals.issubset({0, 1}):
            return False
        return s.nunique(dropna=True) <= 10
    return False

cat_cols = [c for c in feature_cols if is_categorical(train[c])]
num_cols = [c for c in feature_cols if c not in cat_cols]

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer(
    [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
    remainder="drop",
)
preprocessor.set_output(transform="pandas")

classifier = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=100, random_state=42)
)
model = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
model.set_output(transform="pandas")

X = train[feature_cols].copy()
y = train[targets].copy()
fitted = mlca_runtime.fit_record(model, X, "train", y=y)
saved_path = mlca_runtime.save_model(fitted, "alternative_model.pkl")

print(f"features={len(feature_cols)} numeric={len(num_cols)} categorical={len(cat_cols)}")
print(f"train_rows={len(train)} targets={targets}")
print(f"saved={saved_path.name}")

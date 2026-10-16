import json
import mlca_runtime
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing {key}. Available: {list(protocol)}")

train = pd.read_parquet(paths["train"])
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
drop = set(targets) | set(excluded) | {group}
features = [c for c in train.columns if c not in drop]
features = [c for c in features if not ((train[c].dtype == object or pd.api.types.is_string_dtype(train[c])) and train[c].nunique() == len(train))]

num_cols = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in features if c not in num_cols]
high_card = [c for c in cat_cols if train[c].nunique() > 0.05 * len(train)]
cat_cols = [c for c in cat_cols if c not in high_card]
if high_card:
    print(f"dropped high-cardinality: {high_card}")

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = [c for c in cat_cols if train[c].astype(str).str.strip().isin(placeholders).any()]
if ph_cols:
    print(f"placeholders in {len(ph_cols)} columns")

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
pre = ColumnTransformer(transformers, remainder="drop")

width = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
if width <= 300:
    pre.set_output(transform="pandas")

fitted = mlca_runtime.fit_record(pre, train[features], "train")
mlca_runtime.save_model(fitted, "preprocessor.pkl")
print(f"features={len(features)} num={len(num_cols)} cat={len(cat_cols)}")
print(f"encoded_width={width}")

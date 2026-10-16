import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise ValueError(f"Missing {key} in protocol. Available keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
target = targets[0]

all_cols = list(train.columns)
features = [c for c in all_cols if c not in targets and c not in excluded and c != group]

key_like = []
for c in features:
    if (train[c].dtype == object or pd.api.types.is_string_dtype(train[c])) and train[c].nunique() == len(train):
        key_like.append(c)
    elif pd.api.types.is_integer_dtype(train[c]) and train[c].nunique() == len(train) and any(k in c.lower() for k in ["id", "index"]):
        key_like.append(c)
features = [c for c in features if c not in key_like]

derived = []
for c in features:
    if train[c].equals(train[target]):
        derived.append(c)
        print(f"Excluded feature {c}: exact derivation of target")
features = [c for c in features if c not in derived]

placeholders = ["?", "-", "NA", "N/A"]
placeholder_cols = []
for c in features:
    if train[c].dtype == object or pd.api.types.is_string_dtype(train[c]):
        train[c] = train[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
        val[c] = val[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
        mask = train[c].isin(placeholders) | (train[c] == "")
        if mask.any():
            placeholder_cols.append(c)
            train.loc[mask, c] = np.nan
            val.loc[val[c].isin(placeholders) | (val[c] == ""), c] = np.nan
if placeholder_cols:
    print(f"Placeholders found in {len(placeholder_cols)} columns: {placeholder_cols}")

num_cols = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in features if c not in num_cols]

high_card = [c for c in cat_cols if train[c].nunique() > 0.05 * len(train)]
if high_card:
    print(f"Dropped high-cardinality categorical columns: {high_card}")
    features = [c for c in features if c not in high_card]
    cat_cols = [c for c in cat_cols if c not in high_card]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
num_pipe.set_output(transform="pandas")
cat_pipe.set_output(transform="pandas")

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])
preprocessor.set_output(transform="pandas")

X_train = train[features]
y_train = train[target]
mlca_runtime.fit_record(preprocessor, X_train, "train", y=y_train)

X_val = val[features]
val_trans = preprocessor.transform(X_val)
print(f"Train features: {len(features)}, val transformed shape: {val_trans.shape}")

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

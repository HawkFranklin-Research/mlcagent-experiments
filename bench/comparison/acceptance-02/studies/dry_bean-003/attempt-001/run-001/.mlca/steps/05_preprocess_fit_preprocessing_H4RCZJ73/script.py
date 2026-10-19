import json
import itertools
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
protocol = json.loads(inp["_protocol"].read_text())

targets = list(protocol["targets"])
excluded_columns = list(protocol["excluded_columns"])
group_column = protocol.get("group_column")
target = targets[0] if len(targets) == 1 else targets

def is_keylike(name: str, s: pd.Series) -> bool:
    low = name.lower()
    if pd.api.types.is_object_dtype(s) and s.is_unique:
        return True
    if pd.api.types.is_integer_dtype(s) and s.is_unique and any(k in low for k in ("id", "key", "code", "index")):
        return True
    return False

cols = []
for c in train.columns:
    if c in targets or c in excluded_columns or c == group_column:
        continue
    if is_keylike(c, train[c]):
        continue
    cols.append(c)

excluded_feats = []
if len(targets) == 1:
    for c in list(cols):
        if train[c].equals(train[target]):
            excluded_feats.append((c, "exact copy of target"))
    if pd.api.types.is_numeric_dtype(train[target]):
        y = train[target]
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(train[c]) and c not in {x[0] for x in excluded_feats}]
        for r in (1, 2, 3):
            for combo in itertools.combinations(num_cols, r):
                block = train[list(combo)]
                if (block < 0).any(axis=None):
                    continue
                if block.sum(axis=1).equals(y):
                    for c in combo:
                        if c not in {x[0] for x in excluded_feats}:
                            excluded_feats.append((c, f"non-negative row sum equals target: {combo}"))

excluded_names = {c for c, _ in excluded_feats}
cols = [c for c in cols if c not in excluded_names]

if excluded_feats:
    for c, reason in excluded_feats:
        print(f"excluded feature {c}: {reason}")
else:
    print("no target-derived features excluded")

if len(targets) == 1 and pd.api.types.is_numeric_dtype(train[target]):
    warns = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            corr = train[c].corr(train[target])
            if pd.notna(corr) and abs(corr) >= 0.98:
                warns.append(c)
    if warns:
        print(f"high target correlation warning: {warns}")
else:
    print("correlation warning screen skipped for categorical outcome")

text_cols = [c for c in cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
placeholders = {"?", "-", "NA", "N/A"}
found = []
for c in text_cols:
    s = train[c].dropna().astype(str).str.strip()
    mask = s.isin(placeholders) | (s == "")
    if mask.any():
        found.append(c)
print(f"placeholder-like tokens in text features: {found if found else 'none'}")

num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in cols if c not in num_cols]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

transformers = [("num", numeric_pipe, num_cols)]
if cat_cols:
    transformers.append(("cat", categorical_pipe, cat_cols))

preprocessor = ColumnTransformer(transformers, remainder="drop")
preprocessor.set_output(transform="pandas")

mlca_runtime.fit_record(preprocessor, train[cols], "train", train[target])
val_transformed = preprocessor.transform(val[cols])
print(f"features={len(cols)} transformed_shape={val_transformed.shape}")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

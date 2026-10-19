import json
import re
import itertools
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"Missing {key}; available keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column", "")
target_col = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group and group in train.columns:
    drop.add(group)

def key_like(col: str) -> bool:
    s = train[col]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        nn = s.dropna()
        return len(nn) > 0 and nn.nunique() == len(nn)
    if pd.api.types.is_integer_dtype(s):
        nn = s.dropna()
        name = str(col).lower()
        return len(nn) > 0 and nn.nunique() == len(nn) and any(k in name for k in ("id", "key", "code"))
    return False

features = [c for c in train.columns if c not in drop and not key_like(c)]

for c in list(features):
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        for df in (train, val):
            if c in df.columns:
                df[c + "_days"] = (df[c] - pd.Timestamp("1970-01-01")).dt.days
        features.remove(c)
        features.append(c + "_days")

derived = []
if target_col in train.columns:
    y = train[target_col]
    for c in list(features):
        if len(train) and train[c].astype(str).eq(y.astype(str)).all():
            derived.append((c, "exact target equality"))
    if pd.api.types.is_numeric_dtype(y):
        nums = [c for c in features if pd.api.types.is_numeric_dtype(train[c]) and (train[c].dropna() >= 0).all()]
        for r in (1, 2, 3):
            hit = False
            for combo in itertools.combinations(nums, r):
                if train[list(combo)].sum(axis=1).eq(y).all():
                    for c in combo:
                        derived.append((c, f"sum {combo} equals target"))
                    hit = True
                    break
            if hit:
                break

derived_cols = {c for c, _ in derived}
features = [c for c in features if c not in derived_cols]

num_cols = []
cat_cols = []
for c in features:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        cat_cols.append(c)

n = len(train)
cat_cols = [c for c in cat_cols if train[c].nunique(dropna=True) <= max(5, int(0.05 * n))]
model_features = num_cols + cat_cols
if not model_features:
    raise ValueError("No usable feature columns after screening")

ph_cols = []
for c in model_features:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        stripped = s.astype("string").str.strip()
        if stripped.isin(["?", "-", "NA", "N/A", ""]).any() or (s.astype("string") != stripped).any():
            ph_cols.append(c)

width = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
wide = width > 300

transformers = []
if num_cols:
    transformers.append((
        "num",
        Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
        num_cols,
    ))
if cat_cols:
    transformers.append((
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=wide)),
        ]),
        cat_cols,
    ))

preprocessor = ColumnTransformer(transformers, remainder="drop")
if not wide:
    preprocessor.set_output(transform="pandas")

X_train = train[model_features]
X_val = val[model_features]
mlca_runtime.fit_record(preprocessor, X_train, "train")
X_val_t = preprocessor.transform(X_val)

print(f"train rows={len(train)} model features={len(model_features)} numeric={len(num_cols)} categorical={len(cat_cols)}")
print("derived excluded: " + (", ".join(f"{c} ({r})" for c, r in derived) if derived else "none"))
print("placeholder/whitespace columns: " + (", ".join(ph_cols) if ph_cols else "none"))
print(f"val transformed shape={getattr(X_val_t, 'shape', None)}")

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

import mlca_runtime
import json
import itertools
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
target = targets[0]

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
candidate = [c for c in train.columns if c not in drop]

n = len(train)
feature_cols = []
key_like = []
for c in candidate:
    s = train[c]
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        uniq = set(s.dropna().unique())
        if uniq.issubset({0, 1}):
            feature_cols.append(c)
            continue
        if s.notna().all() and s.nunique() == n:
            key_like.append(c)
            continue
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.notna().all() and s.nunique() == n:
        key_like.append(c)
        continue
    feature_cols.append(c)

def same_series(a, b):
    a = a.where(a.notna(), "__NA__").astype(str)
    b = b.where(b.notna(), "__NA__").astype(str)
    return (a == b).all()

target_s = train[target]
excluded_features = []
for c in list(feature_cols):
    if same_series(train[c], target_s):
        excluded_features.append((c, "exact equality with target"))
        feature_cols.remove(c)

if pd.api.types.is_numeric_dtype(target_s) and pd.api.types.is_numeric_dtype(target_s):
    nonneg = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and (train[c].dropna() >= 0).all()]
    for r in range(1, 4):
        for combo in itertools.combinations(nonneg, r):
            vals = train[list(combo)].fillna(0).sum(axis=1)
            if (vals == target_s).all():
                for c in combo:
                    if c in feature_cols:
                        excluded_features.append((c, f"row sum {combo} equals target"))
                        feature_cols.remove(c)
                break
        else:
            continue
        break

if excluded_features:
    print("Excluded features: " + "; ".join(f"{c} ({r})" for c, r in excluded_features))

if pd.api.types.is_numeric_dtype(target_s):
    corr_warn = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            corr = train[[c, target]].corr().iloc[0, 1]
            if abs(corr) >= 0.98:
                corr_warn.append(c)
    if corr_warn:
        print("Correlation >=0.98 with target: " + ", ".join(corr_warn))

text_cols = [
    c for c in feature_cols
    if pd.api.types.is_object_dtype(train[c])
    or pd.api.types.is_string_dtype(train[c])
    or isinstance(train[c].dtype, pd.CategoricalDtype)
]
tokens = ["?", "-", "NA", "N/A"]
found = {t: [] for t in tokens}
found["whitespace"] = []
for c in text_cols:
    s = train[c].dropna().astype(str).str.strip()
    for t in tokens:
        if (s == t).any() or (t == "NA" and (s.str.upper() == "NA").any()):
            found[t].append(c)
    if (s == "").any():
        found["whitespace"].append(c)

ph_parts = [f"'{t}' in {len(cols)} cols" for t, cols in found.items() if cols]
print("Placeholders found: " + ("; ".join(ph_parts) if ph_parts else "none"))

threshold = max(1, int(0.05 * n))
numeric_cols, cat_cols, high_card, date_cols = [], [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    else:
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            try:
                parsed = pd.to_datetime(s.dropna().head(100), errors="coerce")
                if parsed.notna().mean() > 0.95:
                    date_cols.append(c)
                    continue
            except Exception:
                pass
        if s.nunique(dropna=True) <= threshold:
            cat_cols.append(c)
        else:
            high_card.append(c)

feature_cols = numeric_cols + cat_cols
if high_card:
    print(f"Dropped high-cardinality columns: {len(high_card)}")
if date_cols:
    print(f"Dropped date/datetime columns: {len(date_cols)}")

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_steps = []
if any(found[t] for t in tokens) or found["whitespace"]:
    cat_steps.append(("strip", FunctionTransformer(np.char.strip, feature_names_out="one-to-one")))
for t in tokens:
    if found[t]:
        cat_steps.append((f"ph_{t}", SimpleImputer(missing_values=t, strategy="constant", fill_value=np.nan)))
if found["whitespace"]:
    cat_steps.append(("ph_ws", SimpleImputer(missing_values="", strategy="constant", fill_value=np.nan)))
cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
cat_pipe = Pipeline(cat_steps)

width = len(numeric_cols) + sum(min(train[c].nunique(dropna=True) + 1, n) for c in cat_cols)
use_pandas = width <= 300
if not use_pandas:
    cat_pipe.set_params(onehot__sparse_output=True)

transformers = []
if numeric_cols:
    transformers.append(("num", num_pipe, numeric_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
preprocessor = ColumnTransformer(transformers, remainder="drop")

if use_pandas:
    preprocessor.set_output(transform="pandas")
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")

X_train = train[feature_cols]
y_train = train[target]
fitted = mlca_runtime.fit_record(preprocessor, X_train, "train", y=y_train)
X_val_t = fitted.transform(val[feature_cols])
print(f"Preprocessor fit; train {X_train.shape}, val transformed {getattr(X_val_t, 'shape', None)}")

mlca_runtime.save_model(fitted, "preprocessor.pkl")

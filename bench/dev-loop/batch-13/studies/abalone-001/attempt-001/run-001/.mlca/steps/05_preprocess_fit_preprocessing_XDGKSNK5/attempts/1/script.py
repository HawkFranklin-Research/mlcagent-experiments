import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
protocol = json.loads(inp["_protocol"].read_text())
for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key}; available keys: {list(protocol)}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
task = protocol["task"]

def key_like(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.notna().sum() > 1 and s.nunique(dropna=True) == s.notna().sum()
    if pd.api.types.is_integer_dtype(s):
        vals = s.dropna()
        if len(vals) > 1 and vals.nunique() == len(vals):
            name = str(s.name).lower()
            if any(t in name for t in ("id", "key", "code", "index")):
                return True
            if vals.is_monotonic_increasing and (vals.diff().dropna() == 1).all():
                return True
    return False

excl = set(targets) | set(excluded)
if group is not None:
    excl.add(group)
features = [c for c in train.columns if c not in excl]
features = [c for c in features if not key_like(train[c])]

def same(a, b):
    return len(a) == len(b) and np.allclose(a.astype(float), b.astype(float), rtol=0, atol=0, equal_nan=True)

derived = {}
for target in targets:
    if target not in train:
        continue
    y = train[target]
    for c in features:
        if train[c].equals(y) or (
            pd.api.types.is_numeric_dtype(train[c])
            and pd.api.types.is_numeric_dtype(y)
            and same(train[c], y)
        ):
            derived[c] = f"equals {target}"
    if pd.api.types.is_numeric_dtype(y):
        num_all = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
        for r in (1, 2, 3):
            for combo in combinations(num_all, r):
                if all(train[c].notna().all() and (train[c] >= 0).all() for c in combo):
                    if same(train[list(combo)].sum(axis=1), y):
                        for c in combo:
                            derived.setdefault(c, f"sum of {combo} equals {target}")

if derived:
    print("Excluded target derivations:", {c: derived[c] for c in derived})
features = [c for c in features if c not in derived]

warn = []
for target in targets:
    if target not in train or not pd.api.types.is_numeric_dtype(train[target]):
        continue
    for c in features:
        if pd.api.types.is_numeric_dtype(train[c]):
            corr = train[c].corr(train[target])
            if pd.notna(corr) and abs(corr) >= 0.98:
                warn.append(f"{c} ({corr:.2f})")
if warn:
    print("High correlation warnings (kept):", ", ".join(sorted(set(warn))))

text_cols = [
    c for c in features
    if train[c].dtype == object
    or pd.api.types.is_string_dtype(train[c])
    or isinstance(train[c].dtype, pd.CategoricalDtype)
]
ph = {}
for c in text_cols:
    s = train[c].astype("string")
    n = int(s.str.strip().isin(["?", "-", "NA", "N/A", ""]).fillna(False).sum())
    if n:
        ph[c] = n
if ph:
    print(f"Placeholders found in {len(ph)} columns:", ph)

num_cols, cat_cols, high_card = [], [], []
for c in features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        train[c] = (s - pd.Timestamp("1970-01-01")).dt.days.astype(float)
        num_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=True) > 0.05 * len(train):
            high_card.append(c)
        else:
            cat_cols.append(c)
    else:
        high_card.append(c)

if high_card:
    print("Dropped high-cardinality/non-modelable columns:", high_card)

transformers = []
if num_cols:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    num_pipe.set_output(transform="pandas")
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    cat_pipe = Pipeline([
        ("nan_to_empty", SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="")),
        ("strip", FunctionTransformer(np.char.strip, feature_names_out="one-to-one")),
        ("q", SimpleImputer(missing_values="?", strategy="constant", fill_value=np.nan)),
        ("dash", SimpleImputer(missing_values="-", strategy="constant", fill_value=np.nan)),
        ("na", SimpleImputer(missing_values="NA", strategy="constant", fill_value=np.nan)),
        ("na2", SimpleImputer(missing_values="N/A", strategy="constant", fill_value=np.nan)),
        ("empty", SimpleImputer(missing_values="", strategy="constant", fill_value=np.nan)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    cat_pipe.set_output(transform="pandas")
    transformers.append(("cat", cat_pipe, cat_cols))

preprocessor = ColumnTransformer(transformers, remainder="drop")
preprocessor.set_output(transform="pandas")
model_features = num_cols + cat_cols
X_train = train[model_features]
mlca_runtime.fit_record(preprocessor, X_train, "train")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
print(f"Saved preprocessor.pkl; X_train={X_train.shape}, numeric={len(num_cols)}, categorical={len(cat_cols)}")

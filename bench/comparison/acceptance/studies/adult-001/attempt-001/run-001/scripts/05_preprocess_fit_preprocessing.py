import mlca_runtime
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in protocol:
        raise KeyError(f"Missing {k}; available: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
print(f"Train {train.shape} Val {val.shape}")


def is_text(s):
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return False
    return True


all_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
feature_cols = []
for c in all_cols:
    s = train[c]
    n = int(s.notna().sum())
    if n > 1 and s.nunique(dropna=True) == n and (is_text(s) or pd.api.types.is_integer_dtype(s)):
        continue
    feature_cols.append(c)

PLACEHOLDERS = {"?", "-", "NA", "N/A"}


def clean_text(s):
    s2 = s.astype("string").str.strip()
    ph = s2.isin(PLACEHOLDERS) | (s2.notna() & (s2 == ""))
    return s2.mask(ph, pd.NA), bool(ph.any())


ph_cols = []
for c in [x for x in feature_cols if is_text(train[x])]:
    cleaned, has_ph = clean_text(train[c])
    if has_ph:
        ph_cols.append(c)
    train[c] = cleaned.astype(object).where(cleaned.notna(), np.nan)
    if c in val.columns:
        v_cleaned, _ = clean_text(val[c])
        val[c] = v_cleaned.astype(object).where(v_cleaned.notna(), np.nan)
if ph_cols:
    print(f"Placeholders in {len(ph_cols)} cols: {ph_cols[:5]}")

y = train[targets[0]]
y_bin = y.astype(str).str.strip().str.lower().isin([">50k", "1", "true", "yes", "positive", ">50"]).astype(int)
if y_bin.nunique() < 2:
    y_bin = pd.Series(pd.factorize(y)[0], index=y.index)

excluded_deriv = []
num_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
for c in num_feats:
    try:
        if train[c].astype(float).equals(y_bin.astype(float)):
            excluded_deriv.append(c)
    except Exception:
        pass
nonneg = [c for c in num_feats if train[c].notna().all() and (train[c] >= 0).all()][:12]
for k in (2, 3):
    for combo in itertools.combinations(nonneg, k):
        if train[list(combo)].sum(axis=1).equals(y_bin.astype(float)):
            excluded_deriv.extend(combo)
            break
if excluded_deriv:
    print(f"Excluded derivations: {sorted(set(excluded_deriv))}")
    feature_cols = [c for c in feature_cols if c not in excluded_deriv]

if pd.api.types.is_numeric_dtype(y):
    warns = []
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(train[c]):
            continue
        try:
            r = train[c].astype(float).corr(y_bin.astype(float))
            if r is not None and abs(r) >= 0.98:
                warns.append(c)
        except Exception:
            pass
    if warns:
        print(f"Corr>=0.98 warnings: {warns}")

low_card = max(2, int(0.05 * len(train)))
num_cols_out = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c])]
cat_cols_out = [c for c in feature_cols if is_text(train[c]) and train[c].nunique(dropna=True) <= low_card]

num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
transformers = []
if num_cols_out:
    transformers.append(("num", num_pipe, num_cols_out))
if cat_cols_out:
    transformers.append(("cat", cat_pipe, cat_cols_out))

preprocessor = ColumnTransformer(transformers, remainder="drop")
est = len(num_cols_out) + sum(train[c].nunique() for c in cat_cols_out)
if est <= 300:
    try:
        preprocessor.set_output(transform="pandas")
    except Exception:
        pass

X_train = train[feature_cols]
X_val = val[feature_cols]
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train")
Xv = preprocessor.transform(X_val)
print(f"Encoded train={X_train.shape} val={Xv.shape}")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

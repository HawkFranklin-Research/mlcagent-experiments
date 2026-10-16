import mlca_runtime, json, numpy as np, pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in proto:
        raise KeyError(f"missing {k}; available: {list(proto.keys())}")
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group = proto["group_column"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded) | {group}
feats = [c for c in train.columns if c not in drop]
key_like = []
for c in feats:
    s = train[c]
    if s.nunique(dropna=True) == len(s):
        if pd.api.types.is_integer_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            key_like.append(c)
feats = [c for c in feats if c not in key_like]

excluded_feats = []
for t in targets:
    y = train[t]
    for c in list(feats):
        if train[c].equals(y):
            excluded_feats.append((c, f"exact equality with {t}"))
            feats.remove(c)
    nonneg = [c for c in feats if pd.api.types.is_numeric_dtype(train[c]) and (train[c].dropna() >= 0).all()]
    for r in range(1, 4):
        for combo in combinations(nonneg, r):
            if train[list(combo)].sum(axis=1).equals(y):
                for c in combo:
                    if c in feats:
                        excluded_feats.append((c, f"row sum equals {t}"))
                        feats.remove(c)
for c, r in excluded_feats:
    print(f"excluded {c}: {r}")

for t in targets:
    y = train[t]
    for c in feats:
        if pd.api.types.is_numeric_dtype(train[c]):
            r = abs(train[c].corr(y))
            if r >= 0.98:
                print(f"warn: {c} corr {r:.3f} with {t}")

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols, found_ph = [], set()
for c in feats:
    if not pd.api.types.is_numeric_dtype(train[c]):
        v = train[c].dropna().astype(str).str.strip()
        hit = v[v.isin(placeholders)]
        if len(hit):
            ph_cols.append(c)
            found_ph |= set(hit.unique())
if ph_cols:
    print(f"placeholders {sorted(found_ph)} in {len(ph_cols)} columns")

n = len(train)
numeric, low_card, high_card, date_cols = [], [], [], []
for c in feats:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        numeric.append(c)
    elif pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    else:
        try:
            pd.to_datetime(s.dropna().head(20))
            date_cols.append(c)
            continue
        except Exception:
            pass
        nu = s.nunique(dropna=True)
        if nu <= max(1, int(0.05 * n)):
            low_card.append(c)
        else:
            high_card.append(c)

def clean_text(X):
    X = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str).str.strip().replace(
                {"?": np.nan, "-": np.nan, "NA": np.nan, "N/A": np.nan, "": np.nan, "nan": np.nan})
    return X

def date_feats(X):
    out = pd.DataFrame(index=X.index)
    for c in X.columns:
        d = pd.to_datetime(X[c], errors="coerce")
        out[c + "_hour"] = d.dt.hour
        out[c + "_dow"] = d.dt.dayofweek
        out[c + "_month"] = d.dt.month
    return out

def date_names(ft, X):
    return [f"{c}_{p}" for c in X.columns for p in ["hour", "dow", "month"]]

width = len(numeric) + sum(train[c].nunique(dropna=True) for c in low_card) + 3 * len(date_cols)
use_pandas = width <= 300
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=not use_pandas)

transformers = []
if numeric:
    transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                           ("sc", StandardScaler())]), numeric))
if low_card:
    transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                          ("oh", ohe)]), low_card))
if date_cols:
    transformers.append(("date", Pipeline([("dt", FunctionTransformer(date_feats, feature_names_out=date_names)),
                                           ("imp", SimpleImputer(strategy="median")),
                                           ("sc", StandardScaler())]), date_cols))

ct = ColumnTransformer(transformers, remainder="drop")
pre = Pipeline([("clean", FunctionTransformer(clean_text, feature_names_out="one-to-one")), ("ct", ct)])
if use_pandas:
    pre.set_output(transform="pandas")

mlca_runtime.fit_record(pre, train[feats], "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
print(f"features={len(feats)} numeric={len(numeric)} low_card={len(low_card)} high_card={len(high_card)} date={len(date_cols)}")

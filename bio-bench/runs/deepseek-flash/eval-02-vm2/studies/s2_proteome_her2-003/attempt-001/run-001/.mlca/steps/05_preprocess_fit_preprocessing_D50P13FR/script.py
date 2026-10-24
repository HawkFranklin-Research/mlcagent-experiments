import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing keys {missing}. Available top-level keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

exclude_set = set(targets) | excluded
if group_col:
    exclude_set.add(group_col)
candidates = [c for c in train.columns if c not in exclude_set]


def is_keylike(s):
    if len(s) <= 1 or s.nunique(dropna=True) != len(s):
        return False
    return s.dtype.kind in ("i", "u") or s.dtype == object


features = [c for c in candidates if not is_keylike(train[c])]

if len(targets) == 1:
    tgt = train[targets[0]]
    kept = []
    for c in features:
        s = train[c]
        try:
            if s.dtype == tgt.dtype and (s == tgt).all():
                print(f"Excluded '{c}': exact equality with target")
                continue
        except Exception:
            pass
        kept.append(c)
    features = kept

    if np.issubdtype(tgt.dtype, np.number):
        tgt_f = tgt.astype(float)
        corr_warn, subst = [], []
        for c in features:
            if not np.issubdtype(train[c].dtype, np.number):
                continue
            try:
                r = abs(train[c].astype(float).corr(tgt_f))
            except Exception:
                continue
            if r is not None and r >= 0.98:
                corr_warn.append(c)
            if r is not None and r > 0.5 and not (train[c] < 0).any():
                subst.append(c)
        subst = subst[:100]
        deriv = set()
        for i, a in enumerate(subst):
            for b in subst[i + 1:]:
                if np.allclose(train[a] + train[b], tgt_f, rtol=1e-9):
                    deriv.update([a, b])
        for i, a in enumerate(subst):
            for j, b in enumerate(subst[i + 1:], i + 1):
                for c in subst[j + 1:]:
                    if np.allclose(train[a] + train[b] + train[c], tgt_f, rtol=1e-9):
                        deriv.update([a, b, c])
        for f in deriv:
            print(f"Excluded '{f}': row-sum equals target on training rows")
        features = [f for f in features if f not in deriv]
        if corr_warn:
            print(f"Correlation >=0.98 with target (kept {len(corr_warn)}): {corr_warn[:5]}")

num_cols, low_card, high_card, date_cols = [], [], [], []
for c in features:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c); continue
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c); continue
    if s.dtype == object and s.notna().sum() > 0:
        try:
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().sum() >= 0.8 * s.notna().sum():
                date_cols.append(c); continue
        except Exception:
            pass
    nunique = s.nunique(dropna=True)
    if nunique <= max(5, int(0.05 * len(s))):
        low_card.append(c)
    else:
        high_card.append(c)

if date_cols:
    for c in date_cols:
        ts = pd.to_datetime(train[c], errors="coerce")
        tv = pd.to_datetime(val[c], errors="coerce")
        for part in ("hour", "dayofweek", "month"):
            col = f"{c}_{part}"
            train[col] = getattr(ts.dt, part)
            val[col] = getattr(tv.dt, part)
            num_cols.append(col)
    features = [f for f in features if f not in date_cols]
if high_card:
    print(f"Dropped {len(high_card)} high-cardinality text columns")
    features = [f for f in features if f not in high_card]

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = []
for c in low_card:
    s = train[c]
    if s.dtype == object:
        v = s.dropna().astype(str)
        if v.isin(list(placeholders)).any() or (v.str.strip() != v).any():
            ph_cols.append(c)
if ph_cols:
    print(f"Placeholders/whitespace in {len(ph_cols)} text columns -> NaN")
    for c in ph_cols:
        def clean(x):
            if pd.isna(x):
                return np.nan
            v = str(x).strip()
            return np.nan if v in placeholders else v
        train[c] = train[c].map(clean)
        val[c] = val[c].map(clean)

n_cat = sum(train[c].nunique(dropna=True) for c in low_card) if low_card else 0
est_width = len(num_cols) + n_cat

sparse = est_width > 300
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse))])
transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if low_card:
    transformers.append(("cat", cat_pipe, low_card))
pre = ColumnTransformer(transformers, remainder="drop")
if not sparse:
    pre.set_output(transform="pandas")

mlca_runtime.fit_record(pre, train[features].copy(), "train")
tr_cols = pre.transform(train[features].copy()).shape[1]
va_cols = pre.transform(val[features].copy()).shape[1]
print(f"Preprocessor: train {len(train)} rows -> {tr_cols} cols; val {len(val)} -> {va_cols} cols")

mlca_runtime.save_model(pre, "preprocessor.pkl")

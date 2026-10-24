import mlca_runtime
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {list(d.keys())}")
    return d[k]

def is_text(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

def is_key_like(s):
    if len(s) == 0 or s.nunique(dropna=False) != len(s):
        return False
    if is_text(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        vals = np.sort(s.dropna().unique())
        return len(vals) == len(s) and np.all(np.diff(vals) == 1)
    return False

def parse_dates(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if is_text(s):
        p = pd.to_datetime(s, errors="coerce")
        if p.notna().sum() >= max(1, int(0.9 * s.notna().sum())):
            return p
    return None

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
protocol = json.loads(inputs["_protocol"].read_text())
targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
group = protocol.get("group_column")
task = req(protocol, "task")

if isinstance(excluded, dict):
    for key in ("columns", "excluded_columns", "list"):
        if key in excluded and isinstance(excluded[key], list):
            excluded = excluded[key]
            break
    else:
        raise ValueError(f"excluded_columns dict has no list; keys={list(excluded)}")
excluded = set(excluded)

protected = set(targets) | excluded
if group in train.columns:
    protected.add(group)
feature_cols = [c for c in train.columns if c not in protected]
feature_cols = [c for c in feature_cols if not is_key_like(train[c])]

placeholders = {"?": 0, "-": 0, "NA": 0, "N/A": 0, "whitespace": 0}
ph_cols = set()
for c in list(feature_cols):
    if c not in train.columns:
        continue
    s = train[c]
    if is_text(s):
        stripped = s.astype("string").str.strip()
        for tok in ("?", "-", "NA", "N/A"):
            mask = stripped.eq(tok)
            if mask.any():
                placeholders[tok] += int(mask.sum())
                ph_cols.add(c)
                stripped = stripped.mask(mask)
        mask = stripped.eq("")
        if mask.any():
            placeholders["whitespace"] += int(mask.sum())
            ph_cols.add(c)
            stripped = stripped.mask(mask)
        train[c] = stripped
        if c in val.columns:
            vs = val[c].astype("string").str.strip()
            for tok in ("?", "-", "NA", "N/A"):
                vs = vs.mask(vs.eq(tok))
            val[c] = vs.mask(vs.eq(""))

y_train = train[targets[0]]
deriv = set()
deriv_reason = {}
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(y_train):
        if np.array_equal(s.to_numpy(), y_train.to_numpy(), equal_nan=True):
            deriv.add(c)
            deriv_reason[c] = "equals target"
    elif s.fillna("__NA__").astype(str).eq(y_train.fillna("__NA__").astype(str)).all():
        deriv.add(c)
        deriv_reason[c] = "equals target"

if pd.api.types.is_numeric_dtype(y_train):
    yv = y_train.to_numpy()
    if set(np.unique(yv)) <= {0, 1}:
        cand = []
        for c in feature_cols:
            if c in deriv or not pd.api.types.is_numeric_dtype(train[c]):
                continue
            v = train[c].to_numpy()
            if np.isnan(v).any() or (v < 0).any():
                continue
            if np.all(v[yv == 0] == 0):
                cand.append(c)
        for k in (2, 3):
            if k <= len(cand) <= 40:
                for combo in itertools.combinations(cand, k):
                    if np.allclose(train[list(combo)].sum(axis=1).to_numpy(), yv):
                        deriv.update(combo)
                        for c in combo:
                            deriv_reason[c] = "sum equals target"
                        break

feature_cols = [c for c in feature_cols if c not in deriv]

warn = []
if pd.api.types.is_numeric_dtype(y_train):
    num_for_corr = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
    if num_for_corr:
        corr = train[num_for_corr].corrwith(y_train.astype(float))
        warn = corr[corr.abs() >= 0.98].index.tolist()

final_feature_cols, num_cols, cat_cols = [], [], []
dropped = 0
thr = max(5, int(0.05 * len(train)))
ref = pd.Timestamp("2000-01-01")
for c in feature_cols:
    if c not in train.columns:
        continue
    dt = parse_dates(train[c])
    if dt is not None:
        for df in (train, val):
            d = parse_dates(df[c])
            if d is None:
                d = pd.Series(pd.NaT, index=df.index)
            df[f"{c}_days"] = (d - ref).dt.days
            df[f"{c}_month"] = d.dt.month
            df[f"{c}_dow"] = d.dt.dayofweek
            df[f"{c}_hour"] = d.dt.hour
        for name in (f"{c}_days", f"{c}_month", f"{c}_dow", f"{c}_hour"):
            final_feature_cols.append(name)
            num_cols.append(name)
        continue
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        final_feature_cols.append(c)
        num_cols.append(c)
    elif is_text(s):
        if s.nunique(dropna=True) > thr:
            dropped += 1
        else:
            final_feature_cols.append(c)
            cat_cols.append(c)
    else:
        dropped += 1

transformers = []
if num_cols:
    transformers.append((
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        num_cols,
    ))
if cat_cols:
    transformers.append((
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]),
        cat_cols,
    ))
if not transformers:
    raise ValueError("No usable feature columns after screening")

est_width = len(num_cols) + sum(train[c].nunique(dropna=False) for c in cat_cols)
narrow = est_width <= 300
if cat_cols:
    transformers[1][1].named_steps["onehot"].set_params(sparse_output=not narrow)

preprocessor = ColumnTransformer(
    transformers,
    remainder="drop",
    sparse_threshold=0.3 if narrow else 1.0,
)
if narrow:
    preprocessor.set_output(transform="pandas")

mlca_runtime.fit_record(preprocessor, train[final_feature_cols], "train", y=y_train)
Xt_val = preprocessor.transform(val[final_feature_cols])
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

mlca_runtime.record_metric("n_features", float(len(final_feature_cols)), "train")
mlca_runtime.record_metric("n_val_rows", float(len(val)), "val")

if ph_cols:
    found = ", ".join(f"{k}:{v}" for k, v in placeholders.items() if v)
    print(f"Placeholders in {len(ph_cols)} cols: {found}")
else:
    print("Placeholders: none")
if deriv:
    print("Excluded derivations: " + ", ".join(f"{c} ({deriv_reason[c]})" for c in sorted(deriv)))
if warn:
    print("Correlation >=0.98 kept: " + ", ".join(warn[:10]) + ("..." if len(warn) > 10 else ""))
print(f"Features: {len(final_feature_cols)} ({len(num_cols)} numeric, {len(cat_cols)} cat); dropped {dropped}")
print(f"Fitted preprocessor; val shape {Xt_val.shape}")

import mlca_runtime, json, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in drop]

for c in list(feature_cols):
    s = train[c]
    if s.dtype.kind in "OiU" and s.notna().all() and s.nunique() == len(s):
        feature_cols.remove(c)
        print(f"drop key-like: {c}")

ph = ["?", "-", "NA", "N/A", ""]
ph_cols = [c for c in feature_cols if train[c].dtype == object and
           train[c].astype(str).str.strip().isin(ph).any()]
if ph_cols:
    for c in ph_cols:
        train[c] = train[c].astype(object).str.strip().replace(ph, np.nan)
    print(f"placeholders in {len(ph_cols)} cols")

num_all = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
excl = set()
for tgt in targets:
    if tgt not in train.columns:
        continue
    y = train[tgt]
    for c in num_all:
        if c != tgt and train[c].equals(y):
            excl.add(c); print(f"exclude deriv: {c}=={tgt}")
    nn = [c for c in num_all if c not in excl and train[c].dropna().ge(0).all()]
    for i in range(len(nn)):
        for j in range(i + 1, len(nn)):
            for k in range(j + 1, len(nn)):
                if np.allclose(train[nn[i]] + train[nn[j]] + train[nn[k]], y):
                    excl.update([nn[i], nn[j], nn[k]]); print(f"exclude deriv sum=={tgt}")
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 2:
        for c in num_all:
            if c in excl:
                continue
            r = train[[c, tgt]].corr().iloc[0, 1]
            if abs(r) >= 0.98:
                print(f"warn corr: {c} vs {tgt} r={r:.2f}")

feature_cols = [c for c in feature_cols if c not in excl]
num_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
cat_all = [c for c in feature_cols if c not in num_feats]
thr = max(10, int(0.05 * len(train)))
cat_feats = [c for c in cat_all if train[c].nunique() <= thr]
for c in cat_all:
    if c not in cat_feats:
        print(f"drop high-card: {c}")

transformers = [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                 ("sc", StandardScaler())]), num_feats)]
if cat_feats:
    transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                                               sparse_output=False))]),
                         cat_feats))

pre = ColumnTransformer(transformers, remainder="drop")
pre.set_output(transform="pandas")
pre = mlca_runtime.fit_record(pre, train[feature_cols], "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_numeric", float(len(num_feats)), "train")
mlca_runtime.record_metric("n_categorical", float(len(cat_feats)), "train")
print(f"features={len(feature_cols)} num={len(num_feats)} cat={len(cat_feats)}")

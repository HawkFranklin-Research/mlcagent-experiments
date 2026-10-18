import mlca_runtime, json, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])

targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column', None)

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop]

for c in list(feature_cols):
    s = train[c]
    if s.notna().all() and s.nunique() == len(s) and s.dtype.kind in 'OiU':
        feature_cols.remove(c)
        print(f"drop key-like: {c}")

num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
excl = set()
for tgt in targets:
    if tgt not in train.columns:
        continue
    y = train[tgt]
    for c in num_cols:
        if train[c].equals(y):
            excl.add(c); print(f"exclude deriv: {c}=={tgt}")
    nn = [c for c in num_cols if (train[c].dropna() >= 0).all()]
    for i in range(len(nn)):
        for j in range(i + 1, len(nn)):
            for k in range(j + 1, len(nn)):
                s = train[nn[i]] + train[nn[j]] + train[nn[k]]
                if np.allclose(s, y):
                    excl.update([nn[i], nn[j], nn[k]])
                    print(f"exclude deriv: sum=={tgt}")
    if pd.api.types.is_numeric_dtype(y):
        for c in num_cols:
            r = train[[c, tgt]].corr().iloc[0, 1]
            if abs(r) >= 0.98:
                print(f"warn corr: {c} vs {tgt} r={r:.3f}")

feature_cols = [c for c in feature_cols if c not in excl]
numeric_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
cat_feats = [c for c in feature_cols if c not in numeric_feats]

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
transformers = [('num', num_pipe, numeric_feats)]
if cat_feats:
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                         ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    transformers.append(('cat', cat_pipe, cat_feats))

pre = ColumnTransformer(transformers, remainder='drop')
pre.set_output(transform='pandas')
pre.fit(train[feature_cols])
print(f"features: {len(feature_cols)} num={len(numeric_feats)} cat={len(cat_feats)}")
mlca_runtime.save_model(pre, 'preprocessor.pkl')

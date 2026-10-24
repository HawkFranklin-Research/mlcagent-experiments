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
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
protocol = json.loads(inputs['_protocol'].read_text())

def get_protocol_key(key):
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in _protocol. Available keys: {list(protocol.keys())}")
    return protocol[key]

targets = get_protocol_key('targets')
excluded = get_protocol_key('excluded_columns')
group_col = get_protocol_key('group_column')
task = get_protocol_key('task')

all_cols = set(train.columns)
drop_cols = set(targets) | set(excluded)
if group_col is not None and group_col in all_cols:
    drop_cols.add(group_col)
candidate_features = [c for c in train.columns if c not in drop_cols]

key_like = []
for c in candidate_features:
    s = train[c]
    if s.dtype == 'object' or s.dtype.name == 'category':
        if s.nunique() == len(s):
            key_like.append(c)
    elif s.dtype.kind in 'iu':
        if s.nunique() == len(s) and s.max() - s.min() + 1 == len(s):
            key_like.append(c)
if key_like:
    candidate_features = [c for c in candidate_features if c not in key_like]

excluded_deriv = set()
exclusion_log = []
for t in targets:
    y = train[t]
    for c in candidate_features:
        if c in excluded_deriv:
            continue
        if (train[c] == y).all():
            excluded_deriv.add(c)
            exclusion_log.append(f"{c}=exact({t})")
    non_neg = [c for c in candidate_features if c not in excluded_deriv and (train[c] >= 0).all()]
    for i in range(len(non_neg)):
        for j in range(i+1, len(non_neg)):
            c1, c2 = non_neg[i], non_neg[j]
            if ((train[c1] + train[c2]) == y).all():
                for c in (c1, c2):
                    if c not in excluded_deriv:
                        excluded_deriv.add(c)
                        exclusion_log.append(f"{c}=sum2({t})")
    for i in range(len(non_neg)):
        for j in range(i+1, len(non_neg)):
            for k in range(j+1, len(non_neg)):
                c1, c2, c3 = non_neg[i], non_neg[j], non_neg[k]
                if ((train[c1] + train[c2] + train[c3]) == y).all():
                    for c in (c1, c2, c3):
                        if c not in excluded_deriv:
                            excluded_deriv.add(c)
                            exclusion_log.append(f"{c}=sum3({t})")

final_features = [c for c in candidate_features if c not in excluded_deriv]

numeric_cols = []
cat_cols = []
datetime_cols = []
high_card_drop = []
for c in final_features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        datetime_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
        if s.nunique() > 0.05 * len(train):
            high_card_drop.append(c)
        else:
            cat_cols.append(c)
    else:
        high_card_drop.append(c)

placeholder_cols = 0
placeholders = ['?', '-', 'NA', 'N/A']
for c in cat_cols:
    s = train[c].astype(str).str.strip()
    if any((s == ph).any() for ph in placeholders) or (s == '').any():
        placeholder_cols += 1

transformers = []
if numeric_cols:
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    transformers.append(('num', num_pipe, numeric_cols))
if cat_cols:
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    transformers.append(('cat', cat_pipe, cat_cols))

preprocessor = ColumnTransformer(transformers, remainder='drop')
preprocessor.set_output(transform='pandas')
X_train = train[final_features]
preprocessor.fit(X_train)

mlca_runtime.save_model(preprocessor, 'preprocessor.pkl')

print(f"Candidate features after key-like drop: {len(candidate_features)}")
if exclusion_log:
    print(f"Excluded derivations: {'; '.join(exclusion_log[:5])}{'...' if len(exclusion_log) > 5 else ''}")
if datetime_cols:
    print(f"Dropped datetime columns: {datetime_cols}")
if high_card_drop:
    print(f"Dropped high-cardinality text/code columns: {high_card_drop}")
if placeholder_cols:
    print(f"Placeholder tokens found in {placeholder_cols} categorical columns")
print(f"Preprocessor fitted on {len(final_features)} features and saved to preprocessor.pkl")

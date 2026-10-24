import mlca_runtime
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')
all_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

key_cols = [c for c in all_cols if train[c].nunique() == len(train) and 'id' in c.lower()]
all_cols = [c for c in all_cols if c not in key_cols]

excluded_features = []
reasons = []
for target in targets:
    y = train[target]
    for c in all_cols:
        if train[c].equals(y):
            excluded_features.append(c)
            reasons.append(f"{c} equals target")
    non_neg = [c for c in all_cols if pd.api.types.is_numeric_dtype(train[c]) and (train[c] >= 0).all()]
    for r in range(1, 4):
        for combo in combinations(non_neg, r):
            s = train[list(combo)].sum(axis=1) if len(combo) > 1 else train[combo[0]]
            if s.equals(y):
                for c in combo:
                    if c not in excluded_features:
                        excluded_features.append(c)
                        reasons.append(f"{c} in sum equals target")
                break
        else:
            continue
        break
if excluded_features:
    print(f"Excluded derivations: {list(zip(excluded_features, reasons))}")
    all_cols = [c for c in all_cols if c not in excluded_features]

if len(targets) == 1 and pd.api.types.is_numeric_dtype(train[targets[0]]):
    corr_warn = [c for c in all_cols if pd.api.types.is_numeric_dtype(train[c]) and abs(train[c].corr(train[targets[0]])) >= 0.98]
    if corr_warn:
        print(f"High correlation with target: {corr_warn}")

placeholders = ['?', '-', 'NA', 'N/A']
placeholder_cols = []
for c in all_cols:
    if train[c].dtype == 'object' or train[c].dtype.name == 'category':
        for ph in placeholders:
            if (train[c] == ph).any():
                placeholder_cols.append(c)
                break
        else:
            if train[c].apply(lambda x: isinstance(x, str) and x.strip() == '' and x != '').any():
                placeholder_cols.append(c)
if placeholder_cols:
    print(f"Placeholders found in columns: {placeholder_cols}")

object_cols = [c for c in all_cols if train[c].dtype == 'object' or train[c].dtype.name == 'category']
numeric_cols = [c for c in all_cols if c not in object_cols]
high_card = [c for c in object_cols if train[c].nunique() > 0.05 * len(train)]
if high_card:
    print(f"Dropped high-cardinality text columns: {high_card}")
    object_cols = [c for c in object_cols if c not in high_card]
categorical_cols = object_cols

clean_steps = [(f'impute_{ph}', SimpleImputer(missing_values=ph, strategy='constant', fill_value=np.nan)) for ph in placeholders]
if clean_steps and object_cols:
    cleaner = ColumnTransformer([('clean', Pipeline(clean_steps), object_cols)], remainder='passthrough')
    cleaner.set_output(transform='pandas')
    preprocessor = Pipeline([('cleaner', cleaner)])
else:
    preprocessor = Pipeline([])

n_out = len(numeric_cols) + sum(train[c].nunique() for c in categorical_cols)
main_ct = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                      ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_cols)
], remainder='drop')
if n_out <= 300:
    main_ct.set_output(transform='pandas')

preprocessor = Pipeline([('prep', preprocessor), ('main', main_ct)]) if preprocessor.steps else main_ct
preprocessor = mlca_runtime.fit_record(preprocessor, train[all_cols], 'train', y=train[targets[0]])
val_trans = preprocessor.transform(val[all_cols])
print(f"Preprocessor output shape: {val_trans.shape}")
mlca_runtime.save_model(preprocessor, 'preprocessor.pkl')

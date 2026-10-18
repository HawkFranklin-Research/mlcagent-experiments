import mlca_runtime
import pandas as pd
import pickle
import json
from sklearn.pipeline import Pipeline

def main():
    mlca_runtime.set_seed(42)
    inputs = mlca_runtime.inputs()
    protocol = json.loads(inputs['_protocol'].read_text())

    for key in ['targets', 'excluded_columns', 'task']:
        if key not in protocol:
            raise KeyError(f"Missing key '{key}' in protocol. Available: {list(protocol.keys())}")

    targets = protocol['targets']
    excluded = protocol['excluded_columns']
    group_col = protocol.get('group_column', None)

    train = pd.read_parquet(inputs['train'])
    with open(inputs['05_preprocess_fit_preprocessing_QTKFYNMD/preprocessor.pkl'], 'rb') as f:
        preprocessor = pickle.load(f)

    if hasattr(preprocessor, 'feature_names_in_'):
        feature_cols = list(preprocessor.feature_names_in_)
    else:
        all_cols = train.columns.tolist()
        feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]
        key_like = []
        for c in feature_cols:
            s = train[c]
            if s.dtype == 'object' or s.dtype.name == 'string':
                if s.nunique() == len(s):
                    key_like.append(c)
            elif s.dtype.kind in 'iu':
                if s.nunique() == len(s):
                    key_like.append(c)
        feature_cols = [c for c in feature_cols if c not in key_like]

    for c in feature_cols:
        if c in targets or c in excluded:
            raise ValueError(f"Feature column '{c}' is a target or excluded column.")
        if c not in train.columns:
            raise ValueError(f"Feature column '{c}' not found in training data.")

    X_train = train[feature_cols].copy()

    if hasattr(preprocessor, 'set_output'):
        try:
            preprocessor.set_output(transform='pandas')
        except Exception:
            pass

    if isinstance(preprocessor, Pipeline):
        steps = [(name, step) for name, step in preprocessor.steps if name != 'selector']
        steps.append(('selector', 'passthrough'))
        pipeline = Pipeline(steps)
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('selector', 'passthrough')
        ])

    mlca_runtime.fit_record(pipeline, X_train, 'train')
    mlca_runtime.save_model(pipeline, 'selector_all.pkl')

    print(f"Fitted selector on {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print("Saved selector_all.pkl")

if __name__ == '__main__':
    main()

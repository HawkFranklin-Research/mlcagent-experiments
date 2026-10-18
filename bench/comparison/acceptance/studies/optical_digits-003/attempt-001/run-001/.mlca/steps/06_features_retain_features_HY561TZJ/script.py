import mlca_runtime
import json
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


def main():
    mlca_runtime.set_seed(42)
    inputs = mlca_runtime.inputs()
    protocol = json.loads(inputs["_protocol"].read_text())

    needed = ("targets", "excluded_columns", "group_column", "task")
    missing = [k for k in needed if k not in protocol]
    if missing:
        raise ValueError(f"Missing protocol keys {missing}; available: {sorted(protocol)}")

    targets = set(protocol["targets"] or [])
    excluded = set(protocol["excluded_columns"] or [])
    group = protocol["group_column"]
    group_cols = set(group if isinstance(group, (list, tuple, set)) else [group]) if group is not None else set()

    train = pd.read_parquet(inputs["train"])

    feature_cols = []
    for col in train.columns:
        if col in targets or col in excluded or col in group_cols:
            continue
        s = train[col]
        is_text = (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s.dtype)
            or isinstance(s.dtype, pd.CategoricalDtype)
        )
        is_int_seq = pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if (is_text or is_int_seq) and s.nunique(dropna=False) == len(s):
            continue
        feature_cols.append(col)

    X = train.loc[:, feature_cols].copy()

    selector = FunctionTransformer(func=None, validate=False)
    selector = mlca_runtime.fit_record(selector, X, "train")
    mlca_runtime.save_model(selector, "selector_all.pkl")

    print(f"train rows: {len(X)}")
    print(f"features: {len(feature_cols)}")
    print(f"first/last feature: {feature_cols[0] if feature_cols else None} / {feature_cols[-1] if feature_cols else None}")


if __name__ == "__main__":
    main()

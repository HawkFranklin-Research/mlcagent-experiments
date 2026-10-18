import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())


def required(obj, key, ctx):
    if key not in obj:
        raise KeyError(f"Missing {key!r} in {ctx}; available keys: {list(obj.keys())}")
    return obj[key]


targets = required(protocol, "targets", "_protocol")
excluded = required(protocol, "excluded_columns", "_protocol")
group_col = required(protocol, "group_column", "_protocol")
_ = required(protocol, "task", "_protocol")

df = pd.read_parquet(paths["anuran_calls_uci"])

base_exclude = set(targets) | set(excluded) | {group_col}


def is_key_like(s: pd.Series) -> bool:
    if len(s) == 0 or s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        return True
    return False


features = [c for c in df.columns if c not in base_exclude]
features = [c for c in features if not is_key_like(df[c])]

numeric_cols = [
    c for c in features
    if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
]
categorical_cols = [c for c in features if c not in numeric_cols]
numeric_set = set(numeric_cols)

cols = ["column", "scope", "kind", "n", "n_missing", "mean", "sd", "min",
        "median", "max", "n_levels", "top_share"]
rows = []

for c in features:
    s = df[c]
    miss = int(s.isna().sum())
    if c in numeric_set:
        rows.append({
            "column": c, "scope": "overall", "kind": "numeric",
            "n": int(s.notna().sum()), "n_missing": miss,
            "mean": s.mean(), "sd": s.std(ddof=1), "min": s.min(),
            "median": s.median(), "max": s.max(),
            "n_levels": np.nan, "top_share": np.nan,
        })
    else:
        vc = s.value_counts(dropna=True)
        rows.append({
            "column": c, "scope": "overall", "kind": "categorical",
            "n": int(s.notna().sum()), "n_missing": miss,
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_levels": int(vc.size),
            "top_share": float(vc.iloc[0] / vc.sum()) if vc.size else np.nan,
        })

target = targets[0] if targets else None
if target in df.columns:
    vc = df[target].value_counts(dropna=False)
    if len(rows) + len(vc) <= 50:
        for val, cnt in vc.items():
            rows.append({
                "column": target, "scope": f"target={val}", "kind": "target_class",
                "n": int(cnt), "n_missing": 0,
                "mean": np.nan, "sd": np.nan, "min": np.nan,
                "median": np.nan, "max": np.nan,
                "n_levels": np.nan, "top_share": float(cnt / len(df)),
            })

summary = pd.DataFrame(rows, columns=cols)
if len(summary) > 50:
    summary = summary.iloc[:50].copy()


def round_sig(x, p=4):
    if pd.isna(x):
        return x
    x = float(x)
    if x == 0:
        return 0.0
    return float(f"{x:.{p}g}")


for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[c] = summary[c].map(round_sig)

mlca_runtime.save_table(summary, "distributions.csv", caption="Column distribution summary")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")

print(f"rows={len(df)} features={len(features)}")
print(f"numeric={len(numeric_cols)} categorical={len(categorical_cols)}")
print(f"distributions.csv rows={len(summary)}")

import mlca_runtime

import json
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = (protocol.get("task") or {}).get("targets") or []
target_col = targets[0] if targets else "Diagnosis"

df = pd.read_parquet(inputs["breast_cancer_diagnostic_uci"])

numeric_cols = list(df.select_dtypes(include="number").columns)
categorical_cols = list(df.select_dtypes(exclude="number").columns)


def r4(value):
    try:
        if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(f"{float(value):.4g}")
    except Exception:
        return None


rows = []
for col in numeric_cols:
    s = df[col]
    rows.append({
        "column": col,
        "type": "numeric",
        "group": "all",
        "mean": r4(s.mean()),
        "sd": r4(s.std(ddof=1)),
        "min": r4(s.min()),
        "median": r4(s.median()),
        "max": r4(s.max()),
        "n_missing": int(s.isna().sum()),
        "n_levels": None,
        "top_level_share": None,
    })
for col in categorical_cols:
    s = df[col]
    vc = s.value_counts(dropna=False)
    top_share = float(vc.iloc[0]) / float(len(s)) if len(s) else 0.0
    rows.append({
        "column": col,
        "type": "categorical",
        "group": "all",
        "mean": None,
        "sd": None,
        "min": None,
        "median": None,
        "max": None,
        "n_missing": int(s.isna().sum()),
        "n_levels": int(s.nunique(dropna=False)),
        "top_level_share": r4(top_share),
    })

# One row group per target class (within-class target summary).
if target_col in df.columns:
    for cls, sub in df.groupby(target_col, dropna=False):
        rows.append({
            "column": target_col,
            "type": "categorical",
            "group": str(cls),
            "mean": None,
            "sd": None,
            "min": None,
            "median": None,
            "max": None,
            "n_missing": int(sub[target_col].isna().sum()),
            "n_levels": 1,
            "top_level_share": 1.0,
        })

summary = pd.DataFrame(
    rows,
    columns=[
        "column", "type", "group", "mean", "sd", "min", "median", "max",
        "n_missing", "n_levels", "top_level_share",
    ],
)
if len(summary) > 50:
    summary = summary.head(50)

mlca_runtime.save_table(summary, "distributions.csv", caption="Univariate distribution summary")

mlca_runtime.record_fact_candidate("n_numeric", int(len(numeric_cols)), unit=None, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(len(categorical_cols)), unit=None, source="distributions.csv")

print(f"rows={len(df)} cols={len(df.columns)} numeric={len(numeric_cols)} categorical={len(categorical_cols)}")

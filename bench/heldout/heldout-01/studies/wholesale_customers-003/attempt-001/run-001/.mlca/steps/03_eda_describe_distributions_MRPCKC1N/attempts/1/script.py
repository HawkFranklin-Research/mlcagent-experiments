import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["wholesale_customers_uci"])

targets = set(proto.get("targets", []))
excluded = set(proto.get("excluded_columns", []))
group = proto.get("group_column")
drop = targets | excluded | ({group} if group else set())

feature_cols = [c for c in df.columns if c not in drop]

def is_keylike(s: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(s) and s.notna().all():
        v = s.dropna()
        return len(v) > 0 and v.nunique() == len(v) and (v.max() - v.min() + 1) == len(v)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        v = s.dropna()
        return len(v) > 0 and v.nunique() == len(v)
    return False

feature_cols = [c for c in feature_cols if not is_keylike(df[c])]

rows = []
for c in feature_cols:
    s = df[c]
    n_missing = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        rows.append({
            "variable": c, "kind": "numeric",
            "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "median": s.median(), "max": s.max(),
            "n_missing": n_missing, "n_levels": np.nan, "top_share": np.nan,
        })
    else:
        vc = s.dropna().value_counts()
        top_share = float(vc.iloc[0] / vc.sum()) if len(vc) else np.nan
        rows.append({
            "variable": c, "kind": "categorical",
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_missing": n_missing, "n_levels": int(s.nunique(dropna=True)),
            "top_share": top_share,
        })

summary = pd.DataFrame(rows, columns=[
    "variable", "kind", "mean", "sd", "min", "median", "max",
    "n_missing", "n_levels", "top_share",
])
for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[col] = summary[col].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)

mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distribution summaries without class labels")

n_numeric = int((summary["kind"] == "numeric").sum())
n_categorical = int((summary["kind"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"features={len(feature_cols)} numeric={n_numeric} categorical={n_categorical}")
print(f"rows={len(df)}")
print("saved=distributions.csv")

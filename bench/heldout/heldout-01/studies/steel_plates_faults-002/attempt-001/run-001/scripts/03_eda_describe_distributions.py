import json
import pandas as pd
import mlca_runtime
from pandas.api.types import is_numeric_dtype, is_bool_dtype

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in proto:
        raise KeyError(f"Missing '{key}' in _protocol; available top-level keys: {list(proto.keys())}")

targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto["group_column"]
task = proto["task"]

df = pd.read_parquet(inputs["steel_plates_faults_uci"])

features = [
    c for c in df.columns
    if c not in set(targets)
    and c not in set(excluded)
    and (group is None or c != group)
]

# Drop text/code key-like columns: object/string columns with all distinct values.
for c in list(features):
    if (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])) and df[c].nunique(dropna=True) == len(df):
        features.remove(c)

placeholders = {"?", "-", "NA", "N/A", ""}
placeholder_cols = []
for c in df.columns:
    if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
        s = df[c].dropna().astype(str).str.strip()
        if s.isin(placeholders).any():
            placeholder_cols.append(c)

def r4(x):
    try:
        if pd.isna(x):
            return float("nan")
        return float(f"{x:.4g}")
    except Exception:
        return float("nan")

rows = []
for c in features:
    missing = int(df[c].isna().sum())
    is_num = is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])
    if is_num:
        rows.append({
            "variable": c, "role": "feature", "type": "numeric",
            "n_missing": missing,
            "mean": r4(df[c].mean()), "sd": r4(df[c].std()),
            "min": r4(df[c].min()), "median": r4(df[c].median()), "max": r4(df[c].max()),
            "n_levels": float("nan"), "top_level": float("nan"), "top_share": float("nan"),
        })
    else:
        vc = df[c].value_counts(dropna=True, normalize=True)
        top_level = vc.index[0] if len(vc) else float("nan")
        top_share = r4(vc.iloc[0]) if len(vc) else float("nan")
        rows.append({
            "variable": c, "role": "feature", "type": "categorical",
            "n_missing": missing,
            "mean": float("nan"), "sd": float("nan"), "min": float("nan"),
            "median": float("nan"), "max": float("nan"),
            "n_levels": int(df[c].nunique(dropna=True)),
            "top_level": top_level, "top_share": top_share,
        })

for c in targets:
    if c not in df.columns:
        continue
    vc = df[c].value_counts(dropna=True, normalize=True)
    top_level = vc.index[0] if len(vc) else float("nan")
    top_share = r4(vc.iloc[0]) if len(vc) else float("nan")
    rows.append({
        "variable": c, "role": "target", "type": "categorical",
        "n_missing": int(df[c].isna().sum()),
        "mean": float("nan"), "sd": float("nan"), "min": float("nan"),
        "median": float("nan"), "max": float("nan"),
        "n_levels": int(df[c].nunique(dropna=True)),
        "top_level": top_level, "top_share": top_share,
    })

cols = [
    "variable", "role", "type", "n_missing", "mean", "sd", "min", "median", "max",
    "n_levels", "top_level", "top_share",
]
summary = pd.DataFrame(rows, columns=cols)
summary = summary.head(50)

mlca_runtime.save_table(summary, "distributions.csv", caption="Summary distributions by feature and target role")

n_numeric = int(((summary["role"] == "feature") & (summary["type"] == "numeric")).sum())
n_categorical = int(((summary["role"] == "feature") & (summary["type"] == "categorical")).sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"rows={len(df)} features={len(features)} numeric={n_numeric} categorical={n_categorical}")
if placeholder_cols:
    print(f"placeholder_tokens_in_columns={len(placeholder_cols)}")

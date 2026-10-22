import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

if "targets" not in protocol or "excluded_columns" not in protocol:
    raise ValueError(f"Missing required protocol keys; available: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
if not targets:
    raise ValueError("No targets declared in protocol")

df = pd.read_parquet(paths["anemia"])

feature_cols = []
for c in df.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = df[c]
    key_like = (
        s.nunique(dropna=True) == len(s)
        and (pd.api.types.is_object_dtype(s) or pd.api.types.is_integer_dtype(s))
        and any(t in c.lower() for t in ("id", "key", "code", "index"))
    )
    if not key_like:
        feature_cols.append(c)

def is_num(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s)

def summarize(s, feature, level, typ):
    nm = int(s.isna().sum())
    n = int(s.notna().sum())
    if typ == "numeric":
        v = s.dropna().astype(float)
        return dict(
            feature=feature, level=level, type="numeric", n=n,
            mean=v.mean() if len(v) else np.nan,
            sd=v.std(ddof=1) if len(v) > 1 else np.nan,
            min=v.min() if len(v) else np.nan,
            median=v.median() if len(v) else np.nan,
            max=v.max() if len(v) else np.nan,
            n_missing=nm, n_levels=np.nan, top_share=np.nan,
        )
    vc = s.value_counts(dropna=True)
    return dict(
        feature=feature, level=level, type="categorical", n=n,
        mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
        n_missing=nm, n_levels=int(vc.size),
        top_share=float(vc.iloc[0] / vc.sum()) if vc.size else np.nan,
    )

overall = []
for c in feature_cols:
    overall.append(summarize(df[c], c, "overall", "numeric" if is_num(df[c]) else "categorical"))

target_col = targets[0]
class_rows = []
classes = list(df[target_col].dropna().unique())
if len(classes) <= 10:
    for cls in classes:
        mask = df[target_col] == cls
        for c in feature_cols:
            typ = "numeric" if is_num(df[c]) else "categorical"
            class_rows.append(summarize(df.loc[mask, c], c, f"{target_col}={cls}", typ))

rows = overall + class_rows
if len(rows) > 50:
    rows = overall[:50] + class_rows[: max(0, 50 - len(overall[:50]))]

out = pd.DataFrame(rows)
cols = ["feature", "level", "type", "n", "mean", "sd", "min", "median", "max",
        "n_missing", "n_levels", "top_share"]
out = out.reindex(columns=cols)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[c] = out[c].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)

mlca_runtime.save_table(out, "distributions.csv", caption="Distribution summaries")

n_numeric = sum(is_num(df[c]) for c in feature_cols)
n_categorical = len(feature_cols) - n_numeric
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"rows={len(df)} features={len(feature_cols)}")
print(f"numeric={n_numeric} categorical={n_categorical}")
print(f"saved distributions.csv rows={len(out)}")

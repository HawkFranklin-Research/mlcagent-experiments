import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
try:
    targets = proto["targets"]
    excluded = proto["excluded_columns"]
except KeyError as e:
    raise KeyError(
        f"Missing {e.args[0]} in _protocol; available keys: {list(proto.keys())}"
    ) from e

df = pd.read_parquet(inputs["heart_disease_cleveland_uci"])
group_col = proto.get("group_column")
features = [c for c in df.columns if c not in targets and c not in excluded]
if group_col:
    features = [c for c in features if c != group_col]
if len(features) > 50:
    features = features[:50]

max_classes = max(0, (50 - len(features)) // max(1, len(features)))
candidates = []
for t in targets:
    ncls = df[t].nunique(dropna=True)
    if 1 < ncls <= max_classes:
        candidates.append((ncls, t))
group_target = min(candidates)[1] if candidates else None

groups = [("overall", df)]
if group_target is not None:
    for v in sorted(df[group_target].dropna().unique()):
        groups.append((f"{group_target}={v}", df[df[group_target] == v]))

rows = []
for gname, gdf in groups:
    for c in features:
        s = gdf[c]
        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        row = {
            "variable": c,
            "kind": "numeric" if is_num else "categorical",
            "group": gname,
            "n_missing": int(s.isna().sum()),
            "mean": np.nan,
            "sd": np.nan,
            "min": np.nan,
            "median": np.nan,
            "max": np.nan,
            "n_levels": np.nan,
            "top_level": np.nan,
            "top_share": np.nan,
        }
        if is_num:
            row.update(
                {
                    "mean": s.mean(),
                    "sd": s.std(),
                    "min": s.min(),
                    "median": s.median(),
                    "max": s.max(),
                }
            )
        else:
            vc = s.value_counts(dropna=True)
            row["n_levels"] = int(vc.size)
            if vc.size:
                row["top_level"] = str(vc.index[0])
                row["top_share"] = float(vc.iloc[0] / vc.sum())
        rows.append(row)

cols = [
    "variable", "kind", "group", "n_missing", "mean", "sd",
    "min", "median", "max", "n_levels", "top_level", "top_share",
]
dist = pd.DataFrame(rows, columns=cols)

def sig4(x):
    if pd.isna(x):
        return x
    try:
        return float(f"{x:.4g}")
    except Exception:
        return x

for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    dist[c] = dist[c].map(sig4)
dist["n_levels"] = pd.to_numeric(dist["n_levels"], errors="coerce")

overall = dist[dist["group"] == "overall"]
n_numeric = int((overall["kind"] == "numeric").sum())
n_categorical = int((overall["kind"] == "categorical").sum())

mlca_runtime.save_table(dist, "distributions.csv", caption="Column distribution summary")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"rows={len(dist)}, cols={dist.shape[1]}")
print(f"numeric={n_numeric}, categorical={n_categorical}")
print(f"group_target={group_target}")

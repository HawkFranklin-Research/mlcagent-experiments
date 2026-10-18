import mlca_runtime, json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["covertype_uci"])

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
target = targets[0]

feat_cols = [c for c in df.columns if c not in set(targets) | set(excluded) and c != group_col]
key_like = []
for c in feat_cols:
    s = df[c]
    if pd.api.types.is_integer_dtype(s) and s.notna().all() and s.nunique() == len(s) and s.nunique() > 100:
        key_like.append(c)
feat_cols = [c for c in feat_cols if c not in key_like]

text_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(df[c])]
placeholders = {"?", "-", "NA", "N/A", "na", "n/a"}
found = {}
for c in text_cols:
    sv = df[c].astype(str).str.strip()
    mask = sv.isin(placeholders) | (sv.str.len() == 0)
    if mask.any():
        found[c] = int(mask.sum())
print(f"text_placeholder_columns={len(found)}")

wilderness = [c for c in feat_cols if c.startswith("Wilderness_Area")]
soil = [c for c in feat_cols if c.startswith("Soil_Type")]
continuous = [c for c in feat_cols if c not in wilderness and c not in soil]


def num_stats(s):
    return dict(mean=float(s.mean()),
                sd=float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan,
                min=float(s.min()), median=float(s.median()),
                max=float(s.max()), n_missing=int(s.isna().sum()))


def cat_stats(s):
    vc = s.value_counts(dropna=True)
    return dict(n_levels=int(s.nunique(dropna=True)),
                top_share=float(vc.iloc[0] / vc.sum()) if len(vc) else np.nan)


rows = []
for c in continuous:
    rows.append(dict(feature=c, kind="numeric", group="all", **num_stats(df[c])))

focus = ["Elevation", "Horizontal_Distance_To_Hydrology",
         "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"]
focus = [c for c in focus if c in continuous]
for c in focus:
    for cls, sub in df.groupby(target):
        rows.append(dict(feature=c, kind="numeric", group=f"class_{cls}", **num_stats(sub[c])))

if wilderness:
    rows.append(dict(feature="Wilderness_Area", kind="categorical", group="all",
                     **cat_stats(df[wilderness].idxmax(axis=1))))
if soil:
    rows.append(dict(feature="Soil_Type", kind="categorical", group="all",
                     **cat_stats(df[soil].idxmax(axis=1))))
rows.append(dict(feature=target, kind="categorical", group="all", **cat_stats(df[target])))

cols = ["feature", "kind", "group", "mean", "sd", "min", "median", "max",
        "n_missing", "n_levels", "top_share"]
out = pd.DataFrame(rows).reindex(columns=cols)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[c] = out[c].apply(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)
out["n_missing"] = out["n_missing"].astype("Int64")
out["n_levels"] = out["n_levels"].astype("Int64")
out = out.head(50)

mlca_runtime.save_table(out, "distributions.csv",
    caption="Per-feature distribution summary with per-class group means for key continuous features.")

n_num = int(out[(out["kind"] == "numeric") & (out["group"] == "all")]["feature"].nunique())
n_cat = int(out[(out["kind"] == "categorical") & (out["group"] == "all") & (out["feature"] != target)]["feature"].nunique())
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")
print(f"rows={len(out)} numeric_features={n_num} categorical_features={n_cat}")

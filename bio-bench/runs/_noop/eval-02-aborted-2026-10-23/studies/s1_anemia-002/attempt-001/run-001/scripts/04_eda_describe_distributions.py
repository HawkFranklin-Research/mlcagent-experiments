import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
if "targets" not in proto:
    raise KeyError(f"Missing 'targets'; available keys: {list(proto.keys())}")
df = pd.read_parquet(inputs["anemia"])
targets = proto["targets"]
excluded = proto.get("excluded_columns", []) or []
group = proto.get("group_column")
target = targets[0]

feat = [c for c in df.columns if c not in targets and c not in excluded and c != group]
keep = []
for c in feat:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        s2 = s.dropna()
        if len(s2) > 5 and s2.nunique() == len(s2):
            continue
    elif pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        s2 = s.dropna()
        if len(s2) > 5 and s2.nunique() == len(s2):
            continue
    keep.append(c)
feat = keep

num_cols = [c for c in feat if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
cat_cols = [c for c in feat if c not in num_cols]

rows = []
def add_group(gname, mask):
    for c in num_cols:
        s = df.loc[mask, c]
        rows.append({"variable": c, "type": "numeric", "group": gname,
                     "mean": s.mean(), "sd": s.std(), "min": s.min(), "median": s.median(),
                     "max": s.max(), "n_missing": int(s.isna().sum()),
                     "n_levels": np.nan, "top_share": np.nan, "n": int(len(s))})
    for c in cat_cols:
        s = df.loc[mask, c]
        vc = s.value_counts(dropna=True)
        top = vc.iloc[0] if len(vc) else np.nan
        n = int(s.notna().sum())
        rows.append({"variable": c, "type": "categorical", "group": gname,
                     "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
                     "n_missing": int(s.isna().sum()), "n_levels": int(s.nunique(dropna=True)),
                     "top_share": (top / n if n else np.nan), "n": n})

add_group("overall", pd.Series(True, index=df.index))
if target in df.columns:
    for val in df[target].dropna().unique():
        add_group(f"{target}={val}", df[target] == val)

res = pd.DataFrame(rows).head(50)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    res[c] = pd.to_numeric(res[c], errors="coerce").round(4)

mlca_runtime.save_table(res, "distributions.csv", caption="Distribution summaries by variable and target class")
mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")
print(f"rows={len(res)} numeric={len(num_cols)} categorical={len(cat_cols)}")

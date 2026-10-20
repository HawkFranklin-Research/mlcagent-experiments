import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol["group_column"]
df = pd.read_parquet(inputs["wine_quality_uci"])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in df.columns if c not in drop]

def is_keylike(s):
    n = s.notna().sum()
    if n <= 1:
        return False
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == n
    return False

feature_cols = [c for c in feature_cols if not is_keylike(df[c])]
numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in feature_cols if c not in numeric_cols]
rows = []

def add_num(c, s, group):
    rows.append({
        "variable": c, "type": "numeric", "group": group, "n": int(s.notna().sum()),
        "mean": s.mean(), "sd": s.std(), "min": s.min(), "median": s.median(),
        "max": s.max(), "n_missing": int(s.isna().sum()), "n_levels": np.nan, "top_share": np.nan,
    })

def add_cat(c, s, group):
    vc = s.value_counts(dropna=True)
    top_share = vc.iloc[0] / s.notna().sum() if len(vc) and s.notna().sum() else np.nan
    rows.append({
        "variable": c, "type": "categorical", "group": group, "n": int(s.notna().sum()),
        "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
        "n_missing": int(s.isna().sum()), "n_levels": int(s.nunique(dropna=True)), "top_share": top_share,
    })

for c in numeric_cols:
    add_num(c, pd.to_numeric(df[c], errors="coerce"), "overall")
for c in cat_cols:
    add_cat(c, df[c], "overall")

if len(targets) == 1 and targets[0] in df.columns:
    tgt = targets[0]
    classes = sorted(df[tgt].dropna().unique())
    remaining = max(0, 50 - len(rows))
    k = min(len(numeric_cols), remaining // max(1, len(classes)))
    for cls in classes:
        sub = df[df[tgt] == cls]
        for c in numeric_cols[:k]:
            add_num(c, pd.to_numeric(sub[c], errors="coerce"), f"{tgt}={cls}")

out = pd.DataFrame(rows).head(50)
stat_cols = ["n", "mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_share"]
for col in stat_cols:
    out[col] = out[col].map(lambda x: x if pd.isna(x) or x == 0 else float(f"{x:.4g}"))
out = out[["variable", "type", "group", "n", "mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_share"]]

mlca_runtime.save_table(out, "distributions.csv", caption="Distribution summaries overall and by target class.")
n_num = int(out.loc[out["type"] == "numeric", "variable"].nunique())
n_cat = int(out.loc[out["type"] == "categorical", "variable"].nunique())
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")

print(f"features: {n_num} numeric, {n_cat} categorical")
print(f"rows: {len(out)}; saved distributions.csv")

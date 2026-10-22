import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["anemia"])
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column", None)
target_col = targets[0]

drop_cols = set(targets) | set(excluded)
if group_col:
    drop_cols.add(group_col)

feature_cols = []
for c in df.columns:
    if c in drop_cols:
        continue
    s = df[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        if s.nunique(dropna=True) == len(s):
            continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
        vals = np.sort(s.dropna().to_numpy())
        if len(vals) > 1 and np.all(np.diff(vals) == 1):
            continue
    feature_cols.append(c)

placeholders = {"?", "-", "NA", "N/A", ""}
n_placeholder_cols = 0
for c in feature_cols:
    if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
        stripped = df[c].astype("string").str.strip()
        mask = stripped.isin(placeholders)
        if mask.any():
            n_placeholder_cols += 1
        df[c] = stripped.mask(mask, np.nan)

num_cols, cat_cols = [], []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
        num_cols.append(c)
    elif pd.api.types.is_datetime64_any_dtype(df[c]):
        df[c] = (df[c] - df[c].min()).dt.days
        num_cols.append(c)
    else:
        cat_cols.append(c)

def summarize(frame, label):
    rows = []
    for c in num_cols:
        s = pd.to_numeric(frame[c], errors="coerce")
        rows.append({
            "variable": c, "summary_group": label, "dtype": "numeric",
            "n": int(s.notna().sum()), "n_missing": int(s.isna().sum()),
            "mean": s.mean(), "sd": s.std(), "min": s.min(), "median": s.median(), "max": s.max(),
            "n_levels": np.nan, "top_share": np.nan,
        })
    for c in cat_cols:
        s = frame[c].astype("string").str.strip().replace("", np.nan)
        vc = s.value_counts(dropna=True)
        n_non = int(s.notna().sum())
        rows.append({
            "variable": c, "summary_group": label, "dtype": "categorical",
            "n": n_non, "n_missing": int(len(frame) - n_non),
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
            "n_levels": int(vc.size), "top_share": (vc.iloc[0] / n_non if n_non and vc.size else np.nan),
        })
    return rows

rows = summarize(df, "overall")
classes = list(pd.Series(df[target_col]).dropna().unique())[:10]
if len(rows) * (1 + len(classes)) <= 50:
    for cls in classes:
        rows += summarize(df[df[target_col] == cls], f"{target_col}={cls}")
rows = rows[:50]

out = pd.DataFrame(rows)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[c] = pd.to_numeric(out[c], errors="coerce").round(4)

mlca_runtime.save_table(out, "distributions.csv", caption="Distribution summaries")
mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")

print(f"features={len(feature_cols)} numeric={len(num_cols)} categorical={len(cat_cols)}")
print(f"placeholder tokens found in {n_placeholder_cols} text columns")
print(f"distributions.csv rows={len(out)} cols={out.shape[1]}")

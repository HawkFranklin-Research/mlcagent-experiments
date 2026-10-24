import mlca_runtime, json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
ins = mlca_runtime.inputs()
protocol = json.loads(ins["_protocol"].read_text())
df = pd.read_parquet(ins["anemia"])

targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)

ignore = set(targets) | set(excluded)
if group:
    ignore.add(group)
features = [c for c in df.columns if c not in ignore]

def is_keylike(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.isna().sum() == 0 and s.nunique() == len(s)
    if pd.api.types.is_integer_dtype(s):
        name = s.name.lower()
        return s.isna().sum() == 0 and s.nunique() == len(s) and any(
            k in name for k in ["id", "key", "code", "index", "row"]
        )
    return False

features = [c for c in features if not is_keylike(df[c])]
num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
cat_cols = [c for c in features if c not in num_cols]

ph_tokens = {'?', '-', 'NA', 'N/A', ''}
ph_cols = 0
for c in cat_cols:
    s = df[c].dropna().astype(str).str.strip()
    if s.isin(ph_tokens).any():
        ph_cols += 1
print(f"Placeholder tokens in {ph_cols} text columns")

def sig4(x):
    if pd.isna(x):
        return np.nan
    return float(f"{x:.4g}")

rows = []
target = targets[0] if targets else None
classes = sorted(df[target].dropna().unique()) if target else []

def add_row(var, s, grp):
    if var in num_cols:
        r = {
            "variable": var, "type": "numeric", "group": grp,
            "n": int(s.notna().sum()), "n_missing": int(s.isna().sum()),
            "mean": sig4(s.mean()), "sd": sig4(s.std()), "min": sig4(s.min()),
            "median": sig4(s.median()), "max": sig4(s.max()),
            "n_levels": np.nan, "top_share": np.nan,
        }
    else:
        vc = s.value_counts(normalize=True, dropna=True)
        r = {
            "variable": var, "type": "categorical", "group": grp,
            "n": int(s.notna().sum()), "n_missing": int(s.isna().sum()),
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
            "n_levels": int(s.nunique(dropna=True)),
            "top_share": sig4(vc.iloc[0]) if len(vc) else np.nan,
        }
    rows.append(r)

for c in features:
    add_row(c, df[c], "overall")

if target and len(classes) > 0 and len(rows) * (1 + len(classes)) <= 50:
    for cls in classes:
        sub = df[df[target] == cls]
        for c in features:
            add_row(c, sub[c], f"{target}={cls}")

out = pd.DataFrame(rows)
if len(out) > 50:
    out = out.head(50)

cols = ["variable", "type", "group", "n", "n_missing", "mean", "sd",
        "min", "median", "max", "n_levels", "top_share"]
out = out[cols]
mlca_runtime.save_table(out, "distributions.csv", caption="Distribution summaries by variable and group")

mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")

print(f"features={len(features)} numeric={len(num_cols)} categorical={len(cat_cols)} rows={len(out)}")

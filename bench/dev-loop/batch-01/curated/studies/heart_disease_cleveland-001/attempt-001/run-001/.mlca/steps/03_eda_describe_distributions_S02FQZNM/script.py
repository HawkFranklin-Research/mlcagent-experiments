import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["heart_disease_cleveland_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

if "targets" not in protocol:
    raise KeyError(f"Missing 'targets' in _protocol. Available keys: {list(protocol.keys())}")
if "excluded_columns" not in protocol:
    raise KeyError(f"Missing 'excluded_columns' in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
target_col = targets[0] if targets else None

text_cols = df.select_dtypes(include=["object", "string"]).columns
placeholders = {"?", "-", "NA", "N/A", ""}
found = {}
for c in text_cols:
    s = df[c].astype("string").str.strip()
    mask = s.isin(placeholders) | s.eq("")
    if mask.any():
        found[c] = int(mask.sum())
        df[c] = s.mask(mask, np.nan)
    else:
        df[c] = s

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]
key_like = []
for c in feature_cols:
    s = df[c]
    if s.nunique(dropna=False) == len(df) and (s.dtype == object or pd.api.types.is_integer_dtype(s)):
        key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

def round_sig(x, sig=4):
    if pd.isna(x) or x == 0:
        return x
    return round(x, -int(np.floor(np.log10(abs(x)))) + (sig - 1))

def add_row(rows, s, col, group):
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        rows.append({
            "variable": col, "type": "numeric", "group": group,
            "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "median": s.median(), "max": s.max(),
            "n_missing": int(s.isna().sum()),
            "n_levels": np.nan, "top_share": np.nan,
        })
    else:
        vc = s.value_counts(dropna=True)
        top_share = vc.iloc[0] / vc.sum() if len(vc) > 0 else np.nan
        rows.append({
            "variable": col, "type": "categorical", "group": group,
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_missing": int(s.isna().sum()),
            "n_levels": int(s.nunique(dropna=True)),
            "top_share": top_share,
        })

rows = []
for c in feature_cols:
    add_row(rows, df[c], c, "overall")

if target_col and target_col in df:
    n_classes = df[target_col].nunique(dropna=True)
    if len(feature_cols) * (1 + n_classes) <= 50:
        for cls, sub in df.groupby(target_col, dropna=True):
            for c in feature_cols:
                add_row(rows, sub[c], c, f"{target_col}={cls}")

dist_df = pd.DataFrame(rows)
cols = ["variable", "type", "group", "mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_share"]
dist_df = dist_df[cols]
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    dist_df[c] = dist_df[c].apply(lambda x: round_sig(x) if pd.notna(x) else x)

mlca_runtime.save_table(dist_df, "distributions.csv", caption="Distribution summaries by variable and target class")

overall = dist_df[dist_df["group"] == "overall"]
n_numeric = overall[overall["type"] == "numeric"]["variable"].nunique()
n_categorical = overall[overall["type"] == "categorical"]["variable"].nunique()
mlca_runtime.record_fact_candidate("n_numeric", int(n_numeric), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(n_categorical), source="distributions.csv")

print(f"Rows: {len(df)}, features: {len(feature_cols)}")
print(f"n_numeric={n_numeric}, n_categorical={n_categorical}")
if found:
    print(f"Placeholders found in {len(found)} columns: {', '.join(found.keys())}")

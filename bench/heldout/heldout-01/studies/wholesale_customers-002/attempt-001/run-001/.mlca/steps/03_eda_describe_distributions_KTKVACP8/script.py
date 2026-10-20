import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["wholesale_customers_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    group_col = protocol["group_column"]
    task = protocol["task"]
except KeyError as e:
    raise ValueError(f"Missing protocol key {e}. Available keys: {list(protocol.keys())}")

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]

drop_cols = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s):
        if s.nunique(dropna=False) == len(df):
            drop_cols.append(c)
    elif pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if s.nunique(dropna=False) == len(df):
            drop_cols.append(c)
feature_cols = [c for c in feature_cols if c not in drop_cols]

placeholder_tokens = {"?", "-", "NA", "N/A", ""}
text_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
ph_count = 0
for c in text_cols:
    df[c] = df[c].astype(str).str.strip()
    mask = df[c].isin(placeholder_tokens) | df[c].eq("")
    if mask.any():
        ph_count += 1
        df.loc[mask, c] = np.nan
print(f"Placeholders found in {ph_count} text columns")

numeric_cols, categorical_cols = [], []
new_cols = {}
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        for part in ["year", "month", "day", "dayofweek"]:
            new_name = f"{c}_{part}"
            df[new_name] = getattr(s.dt, part)
            new_cols[new_name] = "numeric"
        df.drop(columns=[c], inplace=True)
    elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if pd.api.types.is_integer_dtype(s) and set(s.dropna().unique()).issubset({0, 1}):
            numeric_cols.append(c)
        elif s.nunique() <= 10:
            categorical_cols.append(c)
        else:
            numeric_cols.append(c)
    else:
        categorical_cols.append(c)
numeric_cols.extend([k for k, v in new_cols.items() if v == "numeric"])

rows = []
for c in numeric_cols:
    s = df[c]
    rows.append({
        "feature": c, "type": "numeric",
        "mean": s.mean(), "sd": s.std(), "min": s.min(),
        "median": s.median(), "max": s.max(), "n_missing": s.isna().sum(),
        "n_levels": np.nan, "top_share": np.nan,
    })
for c in categorical_cols:
    s = df[c]
    vc = s.value_counts(normalize=True, dropna=True)
    rows.append({
        "feature": c, "type": "categorical",
        "mean": np.nan, "sd": np.nan, "min": np.nan,
        "median": np.nan, "max": np.nan, "n_missing": s.isna().sum(),
        "n_levels": s.nunique(dropna=True),
        "top_share": vc.max() if len(vc) > 0 else np.nan,
    })

dist_df = pd.DataFrame(rows)

def round_sig(x, sig=4):
    if pd.isna(x) or x == 0:
        return x
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    dist_df[col] = dist_df[col].apply(lambda x: round_sig(x) if pd.notna(x) else x)

mlca_runtime.save_table(dist_df, "distributions.csv", caption="Distribution summary of features")

mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")

print(f"Numeric features: {len(numeric_cols)}, Categorical features: {len(categorical_cols)}")
print(f"Rows summarized: {len(dist_df)}")

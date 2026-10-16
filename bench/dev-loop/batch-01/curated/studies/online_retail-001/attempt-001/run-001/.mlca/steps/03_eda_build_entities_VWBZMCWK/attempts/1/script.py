import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    group_col = protocol["group_column"]
    task = protocol["task"]
except KeyError as e:
    raise KeyError(f"Missing key {e} in _protocol. Available top-level keys: {list(protocol.keys())}")

df = pd.read_parquet(inputs["online_retail_uci"])

text_cols = df.select_dtypes(include=["object", "string"]).columns
placeholders = ["?", "-", "NA", "N/A"]
placeholder_cols = []
for col in text_cols:
    s = df[col].astype(str)
    mask = s.str.strip().eq("") | s.isin(placeholders)
    if mask.any():
        placeholder_cols.append(col)
        df[col] = df[col].astype(str).str.strip()
        df.loc[mask, col] = np.nan
print(f"Placeholders found in {len(placeholder_cols)} columns: {placeholder_cols}")

n_before = len(df)
df = df.dropna(subset=[group_col])
print(f"Dropped {n_before - len(df)} rows without entity key {group_col}")

def find_col(frame, candidates):
    for c in candidates:
        if c in frame.columns:
            return c
    return None

quantity_col = find_col(df, ["Quantity", "quantity"])
price_col = find_col(df, ["UnitPrice", "unitprice", "price"])
date_col = find_col(df, ["InvoiceDate", "invoicedate", "date"])
invoice_col = find_col(df, ["InvoiceNo", "invoiceno", "invoice"])
if not all([quantity_col, price_col, date_col, invoice_col]):
    raise ValueError("Could not find required columns for aggregation")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])
df["_value"] = df[quantity_col] * df[price_col]

print("Reversing rows (negative Quantity or UnitPrice) are included in frequency and monetary totals; returns reduce totals.")

last_date = df[date_col].max()
entities = (
    df.groupby(group_col)
    .agg(
        recency_days=(date_col, lambda x: (last_date - x.max()).days),
        frequency=(invoice_col, "nunique"),
        monetary_total=("_value", "sum"),
        monetary_mean=("_value", "mean"),
        total_quantity=(quantity_col, "sum"),
        mean_quantity=(quantity_col, "mean"),
        n_lines=(quantity_col, "size"),
    )
    .reset_index()
)

feature_cols = [c for c in entities.columns if c not in targets + excluded + [group_col]]

out_path = mlca_runtime.output_dir() / "entities.parquet"
entities.to_parquet(out_path, index=False)

mlca_runtime.record_metric("n_entities", len(entities), "eda")
mlca_runtime.record_metric("mean_recency_days", float(entities["recency_days"].mean()), "eda")
mlca_runtime.record_metric("mean_frequency", float(entities["frequency"].mean()), "eda")
mlca_runtime.record_metric("mean_monetary_total", float(entities["monetary_total"].mean()), "eda")

print(f"Entities shape: {entities.shape}")
print(f"Mean recency days: {entities['recency_days'].mean():.2f}")

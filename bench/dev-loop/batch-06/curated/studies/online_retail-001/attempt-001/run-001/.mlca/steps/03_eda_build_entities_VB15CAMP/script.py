import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

proto = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["online_retail_uci"])

group_col = proto.get("group_column") or "CustomerID"
time_col = "InvoiceDate"
if time_col not in df.columns:
    for c in df.columns:
        if "date" in c.lower():
            time_col = c
            break

qty_col = "Quantity" if "Quantity" in df.columns else None
price_col = "UnitPrice" if "UnitPrice" in df.columns else None
inv_col = "InvoiceNo" if "InvoiceNo" in df.columns else None

n0 = len(df)
df = df.dropna(subset=[group_col])
df[group_col] = df[group_col].astype(str).str.strip()
df = df[df[group_col] != ""]

neg = 0
if qty_col is not None:
    neg = int((df[qty_col] < 0).sum())
    df = df[df[qty_col] > 0]

df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.dropna(subset=[time_col])
last_date = df[time_col].max()

if price_col is not None and qty_col is not None:
    df["_value"] = df[qty_col].astype(float) * df[price_col].astype(float)
elif qty_col is not None:
    df["_value"] = df[qty_col].astype(float)
else:
    df["_value"] = 1.0

agg = df.groupby(group_col).agg(
    recency_days=(time_col, lambda s: int((last_date - s.max()).days)),
    frequency=(time_col, "count"),
    total_value=("_value", "sum"),
    mean_value=("_value", "mean"),
)
if inv_col is not None:
    agg["n_invoices"] = df.groupby(group_col)[inv_col].nunique()
agg = agg.reset_index().rename(columns={group_col: "entity_id"})

agg.to_parquet(out / "entities.parquet", index=False)

summary = {
    "n_input_rows": int(n0),
    "n_entities": int(len(agg)),
    "n_reversing_rows_dropped": int(neg),
    "last_invoice_date": str(last_date),
    "aggregation_rule": "drop rows with missing entity key; drop rows with Quantity<=0 (returns/reversals); one row per entity with recency_days, frequency, total_value, mean_value, n_invoices",
}
mlca_runtime.save_json(summary, "entity_summary.json")
mlca_runtime.record_metric("n_entities", float(len(agg)), "train")
mlca_runtime.record_fact_candidate("entity_count", int(len(agg)), source="entities.parquet")

print(f"input rows: {n0}, entities: {len(agg)}")
print(f"reversing rows dropped (Quantity<=0): {neg}")
print(f"last invoice date: {last_date}")
print("aggregation: one row per entity_id with recency_days, frequency, total_value, mean_value, n_invoices")

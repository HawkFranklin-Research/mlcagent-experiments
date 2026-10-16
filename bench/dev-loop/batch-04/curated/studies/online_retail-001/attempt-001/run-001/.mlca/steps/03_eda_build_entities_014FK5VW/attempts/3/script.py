import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out_dir = mlca_runtime.output_dir()

df = pd.read_parquet(inputs["online_retail_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

if "group_column" not in protocol:
    raise KeyError(f"Missing 'group_column' in _protocol. Available keys: {list(protocol.keys())}")
entity_col = protocol["group_column"]

df = df.dropna(subset=[entity_col])
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
last_date = df["InvoiceDate"].max()

if "Quantity" in df.columns and "UnitPrice" in df.columns:
    df["_amount"] = df["Quantity"] * df["UnitPrice"]
else:
    raise KeyError("Expected Quantity and UnitPrice columns to compute amount.")

print("Reversing rows (Quantity < 0) included in net monetary; frequency counts distinct invoices.")

agg = df.groupby(entity_col).agg(
    recency_days=("InvoiceDate", lambda s: (last_date - s.max()).days),
    frequency=("InvoiceNo", "nunique"),
    total_value=("_amount", "sum"),
    mean_value=("_amount", "mean"),
    n_lines=("_amount", "size"),
).reset_index()

agg.to_parquet(out_dir / "entities.parquet", index=False)

mlca_runtime.record_metric("n_entities", float(len(agg)), "train")
mlca_runtime.record_metric("n_input_rows", float(len(df)), "train")

print(f"Input rows: {len(df)}")
print(f"Entities: {len(agg)}")
print(f"Saved entities.parquet with columns: {list(agg.columns)}")

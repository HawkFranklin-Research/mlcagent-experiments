import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out = mlca_runtime.output_dir()
protocol = json.loads(inputs["_protocol"].read_text())

if "group_column" not in protocol:
    raise KeyError(f"missing 'group_column'; available keys: {list(protocol.keys())}")
group_col = protocol["group_column"]
time_cols = protocol["task"]["time_columns"]

df = pd.read_parquet(inputs["online_retail_uci"])
n0 = len(df)

df = df[df[group_col].notna()].copy()
n1 = len(df)

if "Quantity" in df.columns:
    rev = df["Quantity"] < 0
    print(f"Reversing rows (Quantity<0) dropped: {int(rev.sum())}")
    df = df[~rev].copy()

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
ref_time = df[time_cols[0]].max()

agg = df.groupby(group_col).agg(
    frequency=("InvoiceNo", "nunique"),
    monetary_total=("TotalPrice", "sum"),
    monetary_mean=("TotalPrice", "mean"),
    n_lines=("TotalPrice", "size"),
    last_invoice=(time_cols[0], "max"),
).reset_index()

agg["recency_days"] = (ref_time - agg["last_invoice"]).dt.days
agg = agg.rename(columns={"last_invoice": time_cols[0]})
agg = agg[[group_col] + [c for c in agg.columns if c != group_col]]

agg.to_parquet(out / "entities.parquet", index=False)

mlca_runtime.record_metric("n_entities", float(len(agg)), "train")
mlca_runtime.record_metric("n_rows_dropped_no_key", float(n0 - n1), "train")
mlca_runtime.save_json(
    {"n_entities": int(len(agg)), "group_column": group_col, "time_columns": time_cols},
    "entities_summary.json",
)
print(f"entities: {len(agg)}")

import mlca_runtime as rt
import pandas as pd
import json

rt.set_seed(42)
inp = rt.inputs()
proto = json.loads(inp["_protocol"].read_text())
df = pd.read_parquet(inp["online_retail_uci"])

if "group_column" not in proto:
    raise ValueError(f"Missing 'group_column'. Top-level keys: {list(proto.keys())}")
group = proto["group_column"]
if group not in df.columns:
    raise ValueError(f"group_column {group!r} not in data columns: {list(df.columns)}")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
ref_date = df["InvoiceDate"].max()

n0 = len(df)
df = df.dropna(subset=[group])
n_neg = int((df["Quantity"] < 0).sum())
df = df[df["Quantity"] > 0]
print(f"Rule: dropped {n_neg} reversing rows (Quantity<0); {len(df)}/{n0} kept.")

df["line_value"] = df["Quantity"] * df["UnitPrice"]
last = df.groupby(group)["InvoiceDate"].max()
recency = (ref_date - last).dt.days.rename("recency_days")

agg = df.groupby(group).agg(
    frequency=("InvoiceNo", "nunique"),
    n_lines=("InvoiceNo", "size"),
    distinct_stockcodes=("StockCode", "nunique"),
    monetary_total=("line_value", "sum"),
    monetary_mean=("line_value", "mean"),
)
entities = agg.join(recency).reset_index()
for c in ["frequency", "n_lines", "distinct_stockcodes", "monetary_total", "monetary_mean", "recency_days"]:
    entities[c] = entities[c].astype(float)

entities.to_parquet(rt.output_dir() / "entities.parquet", index=False)
rt.record_metric("n_entities", float(len(entities)), "train")
print(f"entities shape: {entities.shape}; group column retained: {group!r}")

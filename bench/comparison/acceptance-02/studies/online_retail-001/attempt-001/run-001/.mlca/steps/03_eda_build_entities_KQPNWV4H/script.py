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

time_cols = (proto.get("task") or {}).get("time_columns") or []
if time_cols and time_cols[0] in df.columns:
    tcol = time_cols[0]
elif "InvoiceDate" in df.columns:
    tcol = "InvoiceDate"
else:
    tcol = None

n0 = len(df)
if "Quantity" in df.columns:
    mask_neg = df["Quantity"] < 0
    n_neg = int(mask_neg.sum())
    df = df[~mask_neg]
else:
    n_neg = 0

df = df.dropna(subset=[group])
print(f"Rule: dropped {n_neg} reversing rows (Quantity<0); {len(df)}/{n0} kept.")

if not {"Quantity", "UnitPrice"}.issubset(df.columns):
    raise ValueError(f"Missing Quantity/UnitPrice. Got: {list(df.columns)}")
df = df.copy()
df["line_value"] = df["Quantity"] * df["UnitPrice"]

if tcol:
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    ref_date = df[tcol].max()

g = df.groupby(group)
entities = pd.DataFrame(index=g.size().index)
entities["frequency"] = g["InvoiceNo"].nunique() if "InvoiceNo" in df.columns else g.size()
entities["n_lines"] = g.size()
if "StockCode" in df.columns:
    entities["distinct_stockcodes"] = g["StockCode"].nunique()
entities["monetary_total"] = g["line_value"].sum()
entities["monetary_mean"] = g["line_value"].mean()
if tcol:
    entities["recency_days"] = (ref_date - g[tcol].max()).dt.days
entities = entities.reset_index()

entities.to_parquet(rt.output_dir() / "entities.parquet", index=False)
rt.save_json({"n_entities": int(len(entities)), "group_column": group,
              "columns": list(entities.columns)}, "summary.json")
rt.record_metric("n_entities", float(len(entities)), "train")
print(f"entities shape: {entities.shape}; group retained: {group!r}")

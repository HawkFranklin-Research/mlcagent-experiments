import json
import numpy as np
import pandas as pd
import mlca_runtime


def req(d, key):
    if key not in d:
        raise KeyError(f"Missing required key {key!r}. Available top-level keys: {sorted(d.keys())}")
    return d[key]


def main():
    mlca_runtime.set_seed(42)
    inputs = mlca_runtime.inputs()
    df = pd.read_parquet(inputs["online_retail_uci"])
    protocol = json.loads(inputs["_protocol"].read_text())

    targets = req(protocol, "targets")
    excluded = req(protocol, "excluded_columns")
    group_col = req(protocol, "group_column")
    task = req(protocol, "task")
    time_cols = list(task.get("time_columns", []))

    placeholders = {"?", "-", "NA", "N/A"}
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    p_total, p_cols = 0, 0
    for c in obj_cols:
        s = df[c].astype("string").str.strip()
        mask = s.isin(placeholders) | s.eq("")
        n = int(mask.sum())
        if n:
            p_total += n
            p_cols += 1
            df[c] = s.mask(mask)

    n_input = len(df)
    df = df.dropna(subset=[group_col]).copy()
    n_dropped = n_input - len(df)
    repeated = bool(df[group_col].duplicated().any())

    amount_col = None
    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df["_amount"] = df["Quantity"] * df["UnitPrice"]
        amount_col = "_amount"
    else:
        for c in df.select_dtypes(include=[np.number]).columns.tolist():
            lc = c.lower()
            if "amount" in lc or "value" in lc or "total" in lc:
                amount_col = c
                break
        if amount_col is not None:
            df["_amount"] = df[amount_col]
            amount_col = "_amount"

    n_return = 0
    if amount_col is not None:
        df["_positive"] = df["_amount"].clip(lower=0)
        df["_negative"] = df["_amount"].clip(upper=0)
        n_return = int((df["_amount"] < 0).sum())

    invoice_col = None
    for cand in ["InvoiceNo", "Invoice", "invoice_no", "invoice"]:
        if cand in df.columns:
            invoice_col = cand
            break
    if invoice_col is None:
        for c in df.columns:
            if "invoice" in c.lower():
                invoice_col = c
                break

    if not time_cols:
        for cand in ["InvoiceDate", "Date", "Timestamp"]:
            if cand in df.columns:
                time_cols = [cand]
                break
        if not time_cols:
            date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if date_cols:
                time_cols = [date_cols[0]]
    for tc in time_cols:
        if tc in df.columns:
            df[tc] = pd.to_datetime(df[tc], errors="coerce")

    grouped = df.groupby(group_col, dropna=False)
    entities = grouped.size().to_frame("frequency")
    if invoice_col is not None:
        entities["distinct_invoices"] = grouped[invoice_col].nunique()
    if amount_col is not None:
        entities["monetary_positive"] = grouped["_positive"].sum()
        entities["monetary_negative"] = grouped["_negative"].sum()
        entities["monetary_net"] = grouped["_amount"].sum()
        entities["monetary_mean"] = grouped["_amount"].mean()
    if "Quantity" in df.columns:
        entities["quantity_total"] = grouped["Quantity"].sum()

    if time_cols and time_cols[0] in df.columns:
        ref = df[time_cols[0]].max()
        last = grouped[time_cols[0]].max()
        entities[time_cols[0]] = last
        entities["recency_days"] = (ref - last).dt.days
        for tc in time_cols[1:]:
            if tc in df.columns:
                entities[tc] = grouped[tc].max()

    entities = entities.reset_index()
    out_dir = mlca_runtime.output_dir()
    entities.to_parquet(out_dir / "entities.parquet", index=False)

    mlca_runtime.record_metric("input_rows", float(n_input), "train")
    mlca_runtime.record_metric("dropped_missing_group", float(n_dropped), "train")
    mlca_runtime.record_metric("entities", float(len(entities)), "train")
    mlca_runtime.record_metric("return_rows", float(n_return), "train")

    print(f"Input rows: {n_input}, group: {group_col}, repeated: {repeated}")
    print(f"Placeholders: {p_cols} columns, {p_total} tokens")
    print(f"Returns: {n_return} rows with negative amount aggregated separately")
    print(f"Entities: {len(entities)}")
    print("Saved entities.parquet")


if __name__ == "__main__":
    main()

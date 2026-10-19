import numpy as np
import pandas as pd
import mlca_runtime


def main():
    mlca_runtime.set_seed(42)
    paths = mlca_runtime.inputs()
    df = pd.read_parquet(paths["anuran_calls_uci"])

    n_rows, n_columns = df.shape
    rows = []
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        pct_missing = np.nan if n_rows == 0 else (n_missing / n_rows) * 100.0
        pct_missing = np.nan if pd.isna(pct_missing) else round(float(pct_missing), 4)
        rows.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "n_unique": n_unique,
                "n_missing": n_missing,
                "pct_missing": pct_missing,
            }
        )

    schema = pd.DataFrame(
        rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"]
    )
    mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

    missing_total = int(schema["n_missing"].sum())
    mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
    mlca_runtime.record_fact_candidate("n_columns", int(n_columns), source="schema.json")
    mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

    print(f"rows={n_rows} cols={n_columns} missing_total={missing_total}")


if __name__ == "__main__":
    main()

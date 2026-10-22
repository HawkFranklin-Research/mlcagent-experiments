import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)


def main():
    paths = mlca_runtime.inputs()
    df = pd.read_parquet(paths["heart_disease_cleveland_uci"])
    protocol = json.loads(paths["_protocol"].read_text())

    n_rows = int(len(df))
    rows = []
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        rows.append(
            {
                "name": str(col),
                "dtype": str(s.dtype),
                "n_unique": int(s.nunique(dropna=True)),
                "n_missing": n_missing,
                "pct_missing": round(n_missing / n_rows * 100.0, 4) if n_rows else float("nan"),
            }
        )

    schema_df = pd.DataFrame(
        rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"]
    )
    mlca_runtime.save_json(schema_df.to_dict(orient="records"), "schema.json", kind="table")

    missing_total = int(schema_df["n_missing"].sum())
    mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
    mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
    mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

    print(f"rows={n_rows} cols={df.shape[1]}")
    print(f"missing_total={missing_total}")
    print("saved schema.json")


if __name__ == "__main__":
    main()

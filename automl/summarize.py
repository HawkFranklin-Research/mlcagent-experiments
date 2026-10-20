"""Summarize all attempts, retaining failures and explicit MLCA metric sources."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def comparison(result):
    mlca = result.get("mlca", {})
    if mlca.get("source") == "shared_scorer":
        primary = mlca["score"]["primary"]
        return primary["name"], primary["value"], "shared_scorer"
    metric = result.get("comparison_metric")
    metrics = [
        m
        for m in result.get("mlca_recorded_metrics", [])
        if metric is None or m["name"] == metric
    ]
    distinct = {(m["name"], m["value"]) for m in metrics}
    if len(distinct) == 1:
        name, value = distinct.pop()
        return name, value, "recorded"
    return None, None, "ambiguous_recorded" if distinct else "mlca_metric_unavailable"


def summarize(root):
    root = Path(root)
    rows = []
    # Supported public result layout: task/study/attempt/framework/result.json.
    for path in sorted(root.glob("*/*/*/*/result.json")):
        result = json.loads(path.read_text())
        metric, baseline, source = comparison(result)
        primary = result.get("test", {}).get("primary", {})
        value = primary.get("value") if result["status"] == "completed" else None
        difference = (
            value - baseline
            if metric == primary.get("name")
            and baseline is not None
            and value is not None
            else None
        )
        rows.append(
            {
                "task": result["task"],
                "run": result.get("run_label", "/".join(path.parts[-4:-2])),
                "framework": result["framework"],
                "metric": primary.get("name", metric),
                "mlca_primary_metric": result.get("mlca_primary_metric"),
                "mlca_test": baseline,
                "mlca_source": source,
                "automl_test": value,
                "difference_automl_minus_mlca": difference,
                "fit_seconds": result.get("fit_seconds"),
                "status": result["status"],
                "error": result.get("message", result.get("reason", "")),
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "run",
        "framework",
        "metric",
        "mlca_primary_metric",
        "mlca_test",
        "mlca_source",
        "automl_test",
        "difference_automl_minus_mlca",
        "fit_seconds",
        "status",
        "error",
    ]
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# AutoML comparison",
        "",
        "Differences are AutoML minus MLCA; lower RMSE is better. Failures are not assigned zero scores.",
        "",
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row)
    for task, items in sorted(groups.items()):
        lines.extend(
            [
                f"## {task}",
                "",
                "| Run | Tool | MLCA (source) | AutoML | Status |",
                "|---|---|---|---|---|",
            ]
        )
        for row in items:
            lines.append(
                f"| {row['run']} | {row['framework']} | {row['mlca_test']} ({row['mlca_source']}) | {row['automl_test']} | {row['status']} |"
            )
        lines.append("")
        for framework in ["autogluon", "flaml"]:
            selected = [r for r in items if r["framework"] == framework]
            matched = [
                r for r in selected if r["difference_automl_minus_mlca"] is not None
            ]
            differences = [r["difference_automl_minus_mlca"] for r in matched]
            median = statistics.median(differences) if differences else "unavailable"
            mlca_median = (
                statistics.median(r["mlca_test"] for r in matched)
                if matched
                else "unavailable"
            )
            automl_median = (
                statistics.median(r["automl_test"] for r in matched)
                if matched
                else "unavailable"
            )
            lines.append(
                f"{framework}: {sum(r['status'] == 'completed' for r in selected)}/{len(selected)} completed; {len(matched)} matched scores; matched medians MLCA={mlca_median}, AutoML={automl_median}; median paired difference: {median}."
            )
        lines.append("")
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    summarize(parser.parse_args().out)

"""List OpenRouter models for the agent-metrics runs: all free text models plus named ones.

Uses the public /models endpoint (no key, no study data). Writes a dated CSV snapshot,
because free models and their providers change over time.

    python list_models.py openai/gpt-6-luna anthropic/claude-haiku-5.5
"""

import csv
import datetime
import json
import sys
import urllib.request
from pathlib import Path

URL = "https://openrouter.ai/api/v1/models?output_modalities=text"


def price(model, key):
    try:
        return float(model["pricing"].get(key) or 0)
    except (TypeError, ValueError):
        return None


def main(selected):
    with urllib.request.urlopen(URL, timeout=30) as response:
        models = json.load(response)["data"]
    rows = []
    for m in models:
        free = price(m, "prompt") == 0 and price(m, "completion") == 0
        if not free and m["id"] not in selected:
            continue
        params = m.get("supported_parameters") or []
        outputs = (m.get("architecture") or {}).get("output_modalities") or []
        index = (m.get("benchmarks") or {}).get("artificial_analysis") or {}
        rows.append(
            {
                "id": m["id"],
                "name": m["name"],
                "group": "free" if free else "selected",
                "text_output": "text" in outputs,
                "router": m["id"] == "openrouter/free",
                "context": m.get("context_length"),
                "usd_per_m_input": round((price(m, "prompt") or 0) * 1e6, 4),
                "usd_per_m_output": round((price(m, "completion") or 0) * 1e6, 4),
                "json_output": "response_format" in params or "structured_outputs" in params,
                "tools": "tools" in params,
                "aa_coding_index": index.get("coding_index"),
                "aa_agentic_index": index.get("agentic_index"),
            }
        )
    missing = [s for s in selected if s not in {r["id"] for r in rows}]
    rows.sort(key=lambda r: (r["group"], -(r["aa_coding_index"] or -1), r["id"]))
    today = datetime.date.today().isoformat()
    out = Path(__file__).parent / f"openrouter_models_{today}.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['group']:<9} {r['id']:<52} coding={r['aa_coding_index']} json={r['json_output']} text={r['text_output']}")
    if missing:
        print("not found:", ", ".join(missing))
    print("snapshot:", out)


if __name__ == "__main__":
    main(sys.argv[1:])

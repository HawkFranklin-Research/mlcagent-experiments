#!/usr/bin/env bash
set -eo pipefail

cd /workspaces/mlcagent-experiments
mkdir -p bench/automl

echo "=== Starting AutoML batch run at $(date -u) ===" >> bench/automl/batch.log

PYTHONPATH=../mlca-revized/src ./automl/.venv/bin/python automl/run_automl.py --batch   --spec automl/specs/development.yaml   --runs-root bench/comparison/acceptance-02   --mlca-python ../mlca-revized/.venv/bin/python   --time-budget 60 --cpus 2 --seed 42   --work-root bench/automl/.work --out bench/automl >> bench/automl/batch.log 2>&1

echo "=== Summarizing results at $(date -u) ===" >> bench/automl/batch.log
./automl/.venv/bin/python automl/summarize.py --out bench/automl >> bench/automl/batch.log 2>&1

echo "=== AutoML batch complete at $(date -u) ===" >> bench/automl/batch.log

# Custom instructions

Verbatim submitted instructions; failures and refusals are retained.

## Instruction 1

{"event": "preprocess", "actor": "user:policy/694bb57b562d", "step": null, "scope": {"layer": 1, "decision": "in_scope", "reason": "Deterministic checks passed"}, "model": null, "outcome": "not achieved", "not_achieved_reason": "Model calls are disabled in brain mode", "word_count": 19}

Before imputation, drop features with more than 20 % missing values, with the fraction computed on training rows only.

## Instruction 2

{"event": "model", "actor": "user:policy/694bb57b562d", "step": null, "scope": {"layer": 1, "decision": "in_scope", "reason": "Deterministic checks passed"}, "model": null, "outcome": "not achieved", "not_achieved_reason": "Model calls are disabled in brain mode", "word_count": 16}

Also fit a linear SVM, a random forest and an elastic-net logistic regression on training rows.

## Instruction 3

{"event": "validate", "actor": "user:policy/694bb57b562d", "step": null, "scope": {"layer": 1, "decision": "in_scope", "reason": "Deterministic checks passed"}, "model": null, "outcome": "not achieved", "not_achieved_reason": "Model calls are disabled in brain mode", "word_count": 27}

On training+validation rows only, repeated stratified 5-fold CV (5 repeats) with the full pipeline inside each fold; report mean PR AUC, ROC AUC and Brier per model.

## Instruction 4

{"event": "validate", "actor": "user:policy/694bb57b562d", "step": null, "scope": {"layer": 1, "decision": "reject", "reason": "Instruction names an undeclared path", "category": "scope_layer1"}, "model": null, "outcome": "not achieved", "not_achieved_reason": null, "word_count": 14}

Calibration curve and decision-curve analysis (net benefit vs treat-all/none, thresholds 0.05–0.95) on validation rows.

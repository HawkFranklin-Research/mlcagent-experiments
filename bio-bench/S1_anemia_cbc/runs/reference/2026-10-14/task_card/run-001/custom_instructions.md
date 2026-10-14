# Custom instructions

Verbatim submitted instructions; failures and refusals are retained.

## Instruction 1

{"event": "eda", "actor": "user:policy/2d792597d4a8", "step": null, "scope": {"layer": 1, "decision": "in_scope", "reason": "Deterministic checks passed"}, "model": null, "outcome": "not achieved", "not_achieved_reason": "Model calls are disabled in brain mode", "word_count": 13}

Pairplot of Hemoglobin, MCV, MCH and MCHC coloured by Result; save one figure.

## Instruction 2

{"event": "model", "actor": "user:policy/2d792597d4a8", "step": null, "scope": {"layer": 1, "decision": "in_scope", "reason": "Deterministic checks passed"}, "model": null, "outcome": "not achieved", "not_achieved_reason": "Model calls are disabled in brain mode", "word_count": 15}

Also fit random forest, RBF-SVM and extra trees on training rows with the same preprocessing.

## Instruction 3

{"event": "validate", "actor": "user:policy/2d792597d4a8", "step": null, "scope": {"layer": 1, "decision": "reject", "reason": "Instruction names an undeclared path", "category": "scope_layer1"}, "model": null, "outcome": "not achieved", "not_achieved_reason": null, "word_count": 23}

Calibration curves (10 bins) for all candidates and a learning curve (F1 vs training size, 5 sizes) for logistic regression, on training/validation rows.

## Instruction 4

{"event": "interpret", "actor": "user:policy/2d792597d4a8", "step": null, "scope": {"layer": 1, "decision": "in_scope", "reason": "Deterministic checks passed"}, "model": null, "outcome": "not achieved", "not_achieved_reason": "Model calls are disabled in brain mode", "word_count": 10}

SHAP summary (beeswarm) for the frozen model on validation rows.

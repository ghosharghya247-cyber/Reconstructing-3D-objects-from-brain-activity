PYTHON ?= python
.PHONY: all check test audit baselines cv train val controls infer-test export report package verify

all:
	$(PYTHON) run_project.py

check:
	$(PYTHON) run_project.py --check-only

test:
	$(PYTHON) -m pytest -q

audit:
	$(PYTHON) -m src.audit --data-dir data

baselines:
	$(PYTHON) -m src.baselines --data-dir data --evaluate

cv:
	$(PYTHON) -m src.cross_validate --data-dir data --folds 5

train:
	$(PYTHON) -m src.train --config configs/best.yaml --save artifacts/best.joblib

val:
	$(PYTHON) -m src.predict --split val --output outputs/val_predictions.npz
	$(PYTHON) evaluate.py outputs/val_predictions.npz --save submission/metrics/val_metrics.json

controls:
	$(PYTHON) -m src.uncertainty --predictions outputs/val_predictions.npz
	$(PYTHON) -m src.controls --repeats 3

infer-test:
	$(PYTHON) -m src.predict --split test --output submission/predictions.npz

export:
	$(PYTHON) -m src.export --predictions submission/predictions.npz --out submission

report:
	$(PYTHON) -m src.report

package:
	$(PYTHON) -m src.package --root submission

verify:
	$(PYTHON) -m src.verify submission --data-dir data
	$(PYTHON) check_submission.py submission/

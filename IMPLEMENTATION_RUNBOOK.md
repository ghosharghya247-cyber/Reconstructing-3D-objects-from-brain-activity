# Implementation runbook

The complete, verified runbook is now [steps.md](steps.md). It covers the actual provided dataset, requirement audit, model transformations, file locations, individual commands, packaged-source execution, validation results and limitations.

```bash
source .venv/bin/activate
python run_project.py --check-only
python run_project.py --reuse-config
python run_project.py
```

The first command checks existing artifacts; the second rebuilds with the selected configuration; the third also reruns grouped model selection. See `outputs/audit/runtime.json` for measurements and `outputs/logs/` for stage output.

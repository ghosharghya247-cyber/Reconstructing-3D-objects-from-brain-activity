# Report generation

The final report is generated automatically from measured results:

```bash
python run_project.py --reuse-config
# Or, once that workflow's measured results and runtime JSON exist:
python -m src.report --out submission/report.pdf
```

`src/report.py` creates exactly three pages: data/method/grouped selection; official validation metrics/uncertainty/brain controls; and qualitative examples/limitations/runtime/citations. Required inputs are described in `steps.md`. It does not fill missing results with placeholders.

Inspect `submission/report.pdf` before sharing. The supplied checker verifies the report exists; page count and content should also be reviewed. The generated report discloses that no pretrained models or external object models were used and distinguishes local validation evidence from the organisers' hidden significance gate.

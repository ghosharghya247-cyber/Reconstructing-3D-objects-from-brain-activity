"""Prepare code/ reproducibly without copying test labels or development outputs."""
import argparse
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="submission")
    p.add_argument("--checkpoint", default="artifacts/best.joblib")
    args = p.parse_args()
    root = Path(args.root)
    code = root / "code"
    code.mkdir(parents=True, exist_ok=True)
    required = ["README.md", "steps.md", "requirements.txt", "requirements.lock.txt", "pytest.ini", "Makefile", "run_project.py", "src", "configs", "tests"]
    supplied = ["starter.py", "evaluate.py", "metrics.py", "check_submission.py"]
    missing = [str(x) for x in required + supplied + [args.checkpoint] if not Path(x).exists()]
    if missing:
        raise FileNotFoundError("Cannot package without: " + ", ".join(missing))
    for file in required + supplied:
        origin = Path(file)
        dest = code / origin.name
        if origin.is_dir():
            shutil.copytree(origin, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(origin, dest)
    checkpoint = Path(args.checkpoint)
    (code / "artifacts").mkdir(exist_ok=True)
    shutil.copy2(checkpoint, code / "artifacts" / "best.joblib")
    metadata = checkpoint.parent / "train_info.json"
    if metadata.is_file():
        shutil.copy2(metadata, code / "artifacts" / "train_info.json")
    evidence = code / "evidence"
    evidence.mkdir(exist_ok=True)
    for directory in ("outputs/audit", "outputs/validation", "outputs/baselines", "outputs/controls"):
        for item in Path(directory).glob("*.json"):
            shutil.copy2(item, evidence / item.name)
    original = Path("outputs/audit/original")
    if original.is_dir():
        shutil.copytree(original, evidence / "original", dirs_exist_ok=True)
    # Older diagnostic-only intervals describe the original model, whose
    # evidence is retained separately rather than mixed with final results.
    (evidence / "diagnostic_uncertainty.json").unlink(missing_ok=True)
    if not (root / "report.pdf").is_file():
        raise FileNotFoundError("Create submission/report.pdf from REPORT_TEMPLATE.md before packaging")
    print("Packaged runnable code. Review required dataset path in submission/code/README.md")

if __name__ == "__main__":
    main()

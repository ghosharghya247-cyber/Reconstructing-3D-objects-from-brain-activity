"""Configuration and official output helpers; no reimplementation of the grader."""
import json
import subprocess
import sys
from pathlib import Path
import joblib
import yaml
from src.model import ModelConfig


def read_config(path):
    with open(path, "r", encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    return ModelConfig(**values)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=lambda obj: obj.item() if hasattr(obj, "item") else str(obj))


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)  # Only load checkpoints you created/trust.


def save_official(path, sample_ids, points):
    try:
        from starter import save_predictions
    except ImportError as e:
        raise RuntimeError("Missing provided starter.py in project root") from e
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(str(path), sample_ids, points)


def evaluate_official(predictions, metrics_file):
    if not Path("evaluate.py").is_file():
        raise RuntimeError("Missing provided evaluate.py in project root")
    metrics_file = Path(metrics_file)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "evaluate.py", str(predictions), "--save", str(metrics_file)], check=True)

"""Train once on training objects with the grouped-CV-selected hyperparameters."""
import argparse
from pathlib import Path
from dataclasses import asdict
from src.dataset import load_split
from src.io_utils import read_config, save_model, write_json
from src.model import FmriShapeRetriever


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--config", default="configs/best.yaml")
    p.add_argument("--save", default="artifacts/best.joblib")
    args = p.parse_args()
    train = load_split(args.data_dir, "train")
    cfg = read_config(args.config)
    model = FmriShapeRetriever(cfg).fit(train["X"], train["group"], train["obj_masks"], train["obj_points"])
    save_model(model, args.save)
    write_json(Path(args.save).parent / "train_info.json", {
        "config": asdict(cfg),
        "train_scans": len(train["X"]),
        "bank_training_objects": len(model.bank_groups_),
        "note": "No validation or test object features used for training"
    })
    print(f"Saved {args.save} | bank objects={len(model.bank_groups_)}")

if __name__ == "__main__":
    main()

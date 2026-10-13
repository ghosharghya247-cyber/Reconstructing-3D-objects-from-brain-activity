"""Generate assignment-compatible prediction NPZ using the provided starter.py."""
import argparse
import numpy as np
from src.dataset import load_split
from src.io_utils import load_model, save_official


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint", default="artifacts/best.joblib")
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    data = load_split(args.data_dir, args.split)
    model = load_model(args.checkpoint)
    points = model.predict(data["X"])
    assert points.shape == (len(data["sample_id"]), 2048, 3)
    assert points.dtype == np.float32
    assert np.isfinite(points).all()
    save_official(args.output, data["sample_id"], points)
    print(f"Saved {args.output}: {points.shape}, dtype={points.dtype}, distinct source shapes={len(np.unique(model.predict_indices(data['X'])))}")

if __name__ == "__main__":
    main()

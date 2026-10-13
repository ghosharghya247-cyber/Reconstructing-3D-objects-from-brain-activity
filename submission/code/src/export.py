"""Use the provided rendering/PLY helper; avoid writing competing formats."""
import argparse
import numpy as np
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="submission/predictions.npz")
    p.add_argument("--out", default="submission")
    args = p.parse_args()
    try:
        from starter import export_outputs
    except ImportError as e:
        raise RuntimeError("Missing starter.py provided in the assignment ZIP") from e
    with np.load(args.predictions, allow_pickle=False) as f:
        ids = f["sample_id"]
        points = f["points"]
    if points.shape != (len(ids), 2048, 3) or points.dtype != np.float32 or not np.isfinite(points).all():
        raise ValueError("Prediction file is malformed")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    export_outputs(args.out, ids, points)
    print(f"Exported {len(ids)} PLY files and {4 * len(ids)} PNG renders via starter.py")

if __name__ == "__main__":
    main()

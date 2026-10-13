"""Independent file integrity checks before running the supplied grader checker."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from src.dataset import load_split
from starter import read_ply


def verify(root, data_dir):
    root = Path(root)
    data = load_split(data_dir, "test")
    with np.load(root / "predictions.npz", allow_pickle=False) as f:
        if set(("points", "sample_id")) - set(f.files):
            raise AssertionError("Missing points or sample_id in predictions.npz")
        ids, pts = f["sample_id"], f["points"]
    if not np.array_equal(ids, data["sample_id"]):
        raise AssertionError("Predicted sample IDs not identical to supplied test IDs in the same order")
    if len(ids) != 167 or pts.shape != (167, 2048, 3):
        raise AssertionError(f"Unexpected predictions shape: {pts.shape}; expected (167,2048,3)")
    if pts.dtype != np.float32 or not np.isfinite(pts).all():
        raise AssertionError("Prediction dtype must be float32 and all values finite")
    ply = sorted((root / "ply").glob("*.ply"))
    render = sorted((root / "renders").glob("*.png"))
    if len(ply) != 167:
        raise AssertionError(f"Expected 167 PLY files; found {len(ply)}")
    if len(render) != 668:
        raise AssertionError(f"Expected 668 renders; found {len(render)}")
    for i, raw_id in enumerate(ids):
        name = str(raw_id)
        if not (root / "ply" / f"{name}.ply").is_file():
            raise AssertionError(f"Missing PLY: {name}")
        exported = read_ply(root / "ply" / f"{name}.ply")
        if exported.shape != (2048, 3) or not np.isfinite(exported).all():
            raise AssertionError(f"Malformed PLY: {name}")
        if not np.allclose(exported, pts[i], rtol=0, atol=1e-5):
            raise AssertionError(f"PLY differs from prediction: {name}")
        for angle in (0, 8, 16, 24):
            png = root / "renders" / f"{name}_view{angle:02d}.png"
            if not png.is_file():
                raise AssertionError(f"Missing render: {png}")
            with Image.open(png) as img:
                if img.size != (256, 256):
                    raise AssertionError(f"Incorrect render size {png}: {img.size}")
    for relative in ("metrics/val_metrics.json", "report.pdf", "code/README.md", "code/steps.md", "code/requirements.txt", "code/run_project.py", "code/artifacts/best.joblib", "code/src/model.py"):
        if not (root / relative).is_file():
            raise AssertionError(f"Missing required submission item: {relative}")
    metrics = json.loads((root / "metrics/val_metrics.json").read_text())
    for key in ("chamfer", "fscore", "silhouette_iou", "silhouette_ssim", "identification", "composite"):
        if key not in metrics or not np.isfinite(metrics[key]):
            raise AssertionError(f"Missing or non-finite metric: {key}")
    if (root / "report.pdf").read_bytes()[:5] != b"%PDF-":
        raise AssertionError("report.pdf is not a PDF")
    print("Independent integrity checks passed: IDs, dtype, shape, PLY/PNG counts and dimensions, report, metrics and code.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="submission")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()
    verify(args.root, args.data_dir)

if __name__ == "__main__":
    main()

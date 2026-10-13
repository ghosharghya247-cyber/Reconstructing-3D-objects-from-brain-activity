"""Inspect split contracts, leakage within a split, numerical ranges, subject shift."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from src.dataset import load_split
from src.io_utils import write_json


def summarize(data, split):
    X = data["X"]
    out = {
        "split": split,
        "scan_count": len(X),
        "X_shape": list(X.shape),
        "X_dtype": str(X.dtype),
        "nonfinite_count": int(np.size(X) - np.isfinite(X).sum()),
        "signal_min": float(np.min(X)),
        "signal_max": float(np.max(X)),
        "signal_mean": float(np.mean(X, dtype=np.float64)),
        "subjects": {str(k): v for k, v in Counter(map(str, data["subject"])).items()},
        "unique_sample_ids": int(len(np.unique(data["sample_id"]))),
    }
    if "group" in data:
        grp, counts = np.unique(data["group"], return_counts=True)
        out.update({
            "unique_objects": len(grp),
            "scans_per_object_min": int(counts.min()),
            "scans_per_object_max": int(counts.max()),
            "shape_points": list(data["obj_points"].shape),
            "shape_masks": list(data["obj_masks"].shape),
            "class_counts_unique_objects": {str(k): int(v) for k, v in zip(*np.unique(data["obj_class"][grp], return_counts=True))},
            "point_coordinate_min": float(np.min(data["obj_points"])),
            "point_coordinate_max": float(np.max(data["obj_points"])),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="outputs/audit/dataset_audit.json")
    args = p.parse_args()
    results = {}
    expected = {"train": (533, 292), "val": (125, 67), "test": (167, None)}
    for split in ("train", "val", "test"):
        data = load_split(args.data_dir, split)
        results[split] = summarize(data, split)
        scans, objects = expected[split]
        if data["X"].shape != (scans, 10, 31141):
            raise ValueError(f"{split}: fMRI shape differs from the assignment: {data['X'].shape}")
        if results[split]["nonfinite_count"]:
            raise ValueError(f"{split}: non-finite fMRI values")
        if objects is not None:
            if len(data["obj_points"]) != objects or results[split]["unique_objects"] != objects:
                raise ValueError(f"{split}: expected {objects} objects")
            if not np.isfinite(data["obj_points"]).all():
                raise ValueError(f"{split}: non-finite object points")
            if data["obj_images"].shape != (objects, 32, 64, 64, 3):
                raise ValueError(f"{split}: invalid stimulus image dimensions")
        elif set(data) != {"X", "subject", "sample_id"}:
            raise ValueError("test must contain only X, subject and sample_id")
        print(f"{split:5s}: {results[split]['scan_count']:4d} scans | shape {data['X'].shape} | subjects {len(results[split]['subjects'])}")
    print("Subjects only in validation:", set(results["val"]["subjects"]) - set(results["train"]["subjects"]))
    print("Subjects only in test:", set(results["test"]["subjects"]) - set(results["train"]["subjects"]))
    with np.load(Path(args.data_dir) / "cameras.npz", allow_pickle=False) as cams:
        if cams["R"].shape != (32, 3, 3) or cams["t"].shape != (32, 3) or int(cams["resolution"]) != 64:
            raise ValueError("Invalid calibrated camera layout")
        for key in ("R", "t", "focal", "cx", "cy"):
            if not np.isfinite(cams[key]).all():
                raise ValueError(f"Non-finite camera field: {key}")
    write_json(args.out, results)
    print("Audit written to", args.out)

if __name__ == "__main__":
    main()

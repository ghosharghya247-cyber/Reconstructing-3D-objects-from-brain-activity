"""Object-clustered intervals for the supplied metrics, with fixed anchors."""
import argparse
import numpy as np
from src.dataset import load_split, object_points_for_scans
from src.io_utils import write_json
from metrics import evaluate, composite, check_predictions, GEOMETRY


def cluster_bootstrap(scores, groups, rng, iterations=2000):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    unique = np.unique(groups)
    by_group = [np.flatnonzero(groups == k) for k in unique]
    means = np.empty(iterations)
    for j in range(iterations):
        selected = rng.integers(0, len(unique), size=len(unique))
        indexes = np.concatenate([by_group[i] for i in selected])
        means[j] = float(scores[indexes].mean())
    return {
        "mean": float(np.mean(scores)),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "units": "object-group clustered scans",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--predictions", default="outputs/val_predictions.npz")
    p.add_argument("--out", default="outputs/validation/official_uncertainty.json")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    val = load_split(args.data_dir, "val")
    with np.load(args.predictions, allow_pickle=False) as f:
        ids = f["sample_id"]
        points = f["points"]
    if not np.array_equal(ids, val["sample_id"]):
        raise ValueError("Prediction IDs/order do not match validation sample IDs")
    gt = object_points_for_scans(val)
    points = check_predictions(points, len(val["X"]))
    with np.load(f"{args.data_dir}/cameras.npz", allow_pickle=False) as z:
        cams = {k: z[k] for k in z.files}
    summary, per = evaluate(points, gt, val["obj_masks"][val["group"]], val["group"], cams)
    with np.load(f"{args.data_dir}/val_anchors.npz", allow_pickle=False) as z:
        generic = {k[8:]: float(z[k]) for k in z.files if k.startswith("generic_")}
        oracle = {k[7:]: float(z[k]) for k in z.files if k.startswith("oracle_")}
    # The supplied composite is affine, so averaging per-scan composites gives
    # precisely the composite of the scan means with these fixed anchors.
    for row in per:
        row["composite"] = composite(row, generic, oracle)
    rng = np.random.default_rng(args.seed)
    result = {
        "method": "95% percentile bootstrap, resampling whole object groups and keeping their scans together",
        "limitations": "Conditional on fixed validation opponent objects and cached generic/oracle anchors; not the organisers' hidden brain-use significance test, not refitted-model uncertainty",
        "iterations": args.iterations,
        "seed": args.seed,
        "metrics": {m: cluster_bootstrap(np.array([r[m] for r in per]), val["group"], rng, args.iterations)
                    for m in GEOMETRY + ["identification", "composite"]},
        "n_object_groups": len(np.unique(val["group"])),
    }
    write_json(args.out, result)
    write_json("outputs/validation/per_scan_metrics.json", {"sample_id": val["sample_id"].tolist(), "group": val["group"].tolist(), "metrics": per})
    print(result)

if __name__ == "__main__":
    main()

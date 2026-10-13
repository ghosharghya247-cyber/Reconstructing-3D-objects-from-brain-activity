"""Grouped CV, with train-fold-only preprocessing and shape descriptor fitting."""
import argparse
import dataclasses
import hashlib
from pathlib import Path
import numpy as np
import yaml
from sklearn.model_selection import GroupKFold
from src.dataset import load_split, group_indices
from metrics import chamfer_and_fscore
from src.io_utils import write_json
from src.model import FmriShapeRetriever, ModelConfig


def shape_metric_cache(points, path="outputs/validation/train_shape_metrics.npz"):
    """Target-only scoring cache; this matrix is never passed into model.fit."""
    path = Path(path)
    digest = hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
    if path.is_file():
        with np.load(path) as z:
            if str(z["sha256"]) == digest:
                return z["chamfer"], z["fscore"]
    n = len(points)
    cd, fs = np.zeros((n, n)), np.eye(n)
    for i in range(n):
        for j in range(i):
            cd[i, j], fs[i, j] = chamfer_and_fscore(points[i], points[j])
            cd[j, i], fs[j, i] = cd[i, j], fs[i, j]
        if i % 25 == 0:
            print(f"Training-shape scoring cache: {i + 1}/{n}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, chamfer=cd, fscore=fs, sha256=digest)
    return cd, fs


def run_cv(train, config, folds=5, scoring_cache=None):
    g = group_indices(train["group"], len(train["obj_points"]))
    n_groups = len(np.unique(g))
    k = min(folds, n_groups)
    if k < 2:
        raise ValueError("Cannot run grouped CV with fewer than two objects")
    cv = GroupKFold(n_splits=k)
    cd, fs = scoring_cache if scoring_cache is not None else shape_metric_cache(train["obj_points"])
    results = []
    for fold, (ix_tr, ix_va) in enumerate(cv.split(train["X"], groups=g), 1):
        assert not set(g[ix_tr]).intersection(set(g[ix_va])), "Group leakage!"
        model = FmriShapeRetriever(config).fit(
            train["X"][ix_tr], g[ix_tr], train["obj_masks"], train["obj_points"]
        )
        predicted_groups = model.predict_train_group(train["X"][ix_va])
        own = cd[predicted_groups, g[ix_va]]
        opponents = np.unique(g[ix_va])
        distances = cd[predicted_groups[:, None], opponents[None, :]]
        # Identical to metrics.evaluate identification: exclude the own object.
        ident = 100 * (own[:, None] < distances).sum(axis=1) / (len(opponents) - 1)
        result = {
            "fold": fold,
            "n_train_scans": len(ix_tr),
            "n_validation_scans": len(ix_va),
            "n_train_groups": len(np.unique(g[ix_tr])),
            "n_validation_groups": len(np.unique(g[ix_va])),
            "diagnostic_chamfer": float(own.mean()),
            "diagnostic_fscore": float(fs[predicted_groups, g[ix_va]].mean()),
            "identification": float(ident.mean()),
        }
        results.append(result)
        print(f"  fold={fold}/{k} Chamfer={result['diagnostic_chamfer']:.5f} F1={result['diagnostic_fscore']:.3f} identification={result['identification']:.2f}%", flush=True)
    return {
        "folds": results,
        "mean_diagnostic_chamfer": float(np.mean([r["diagnostic_chamfer"] for r in results])),
        "mean_diagnostic_fscore": float(np.mean([r["diagnostic_fscore"] for r in results])),
        "mean_identification": float(np.average([r["identification"] for r in results], weights=[r["n_validation_scans"] for r in results])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--candidates", default="configs/candidates.yaml")
    parser.add_argument("--output", default="outputs/validation/cv_results.json")
    parser.add_argument("--best", default="configs/best.yaml")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    train = load_split(args.data_dir, "train")
    with open(args.candidates, encoding="utf-8") as f:
        cfgs = yaml.safe_load(f)
    if not isinstance(cfgs, list) or not cfgs:
        raise ValueError("configs/candidates.yaml must be a non-empty list of config dictionaries")
    all_results = []
    scoring_cache = shape_metric_cache(train["obj_points"])
    for number, cfg in enumerate(cfgs, 1):
        print(f"Candidate {number}/{len(cfgs)}: {cfg}", flush=True)
        model_config = ModelConfig(**cfg)
        summary = run_cv(train, model_config, args.folds, scoring_cache)
        all_results.append({"candidate": number, "config": dataclasses.asdict(model_config), **summary})
    best = max(all_results, key=lambda r: (r["mean_identification"], -r["mean_diagnostic_chamfer"]))
    Path(args.best).parent.mkdir(parents=True, exist_ok=True)
    with open(args.best, "w", encoding="utf-8") as f:
        yaml.safe_dump(best["config"], f, sort_keys=False)
    write_json(args.output, {"model_selection_metric": "grouped_cv_identification_higher_is_better_chamfer_tiebreak", "results": all_results, "best_candidate": best["candidate"]})
    print("Selected candidate", best["candidate"], "| saved", args.best)
    print("Reminder: validation must use supplied evaluate.py, not the CV proxy")

if __name__ == "__main__":
    main()

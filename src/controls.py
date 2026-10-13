"""Test prediction dependence on correct brain/object pairing."""
import argparse
import numpy as np
from src.dataset import load_split
from src.io_utils import load_model, save_official, evaluate_official, write_json
from metrics import chamfer_and_fscore


def pairing_test(correct, val, rng, iterations):
    """Permutation test of object pairing, preserving subject membership."""
    if iterations < 1:
        raise ValueError("permutations must be positive")
    groups = val["group"]
    unique = np.unique(groups)
    rank_by_target = np.empty((len(correct), len(unique)))
    # Repeated retrieved shapes share the same Chamfer distances.
    cache = {}
    for i, cloud in enumerate(correct):
        key = cloud.tobytes()
        if key not in cache:
            distances = np.array([chamfer_and_fscore(cloud, val["obj_points"][o])[0] for o in unique])
            cache[key] = 100 * (distances[:, None] < distances[None, :]).sum(1) / (len(unique) - 1)
        rank_by_target[i] = cache[key]
    targets = np.searchsorted(unique, groups)
    rows = np.arange(len(correct))
    observed = float(rank_by_target[rows, targets].mean())
    blocks = [np.flatnonzero(val["subject"] == s) for s in np.unique(val["subject"])]
    null = np.empty(iterations)
    for j in range(iterations):
        perm = rows.copy()
        for block in blocks:
            perm[block] = rng.permutation(block)
        null[j] = rank_by_target[perm, targets].mean()
    return {
        "method": "one-sided within-subject scan-pairing permutation test, fixed model and validation objects",
        "limitations": "Separate evidence check; does not implement the organisers' unpublished hidden brain-use gate",
        "iterations": iterations,
        "observed_identification": observed,
        "null_mean_identification": float(null.mean()),
        "null_ci95": np.quantile(null, [0.025, 0.975]).tolist(),
        "p_value": float((1 + (null >= observed).sum()) / (iterations + 1)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint", default="artifacts/best.joblib")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--permutations", type=int, default=2000)
    args = p.parse_args()
    val = load_split(args.data_dir, "val")
    model = load_model(args.checkpoint)
    rng = np.random.default_rng(args.seed)
    correct = model.predict(val["X"])
    correct_file = "outputs/controls/correct_val.npz"
    save_official(correct_file, val["sample_id"], correct)
    evaluate_official(correct_file, "outputs/controls/correct_metrics.json")
    rows = []
    for repeat in range(args.repeats):
        perm = rng.permutation(len(val["X"]))
        scrambled = correct[perm]
        path = f"outputs/controls/shuffled_{repeat:02d}.npz"
        metrics = f"outputs/controls/shuffled_{repeat:02d}_metrics.json"
        save_official(path, val["sample_id"], scrambled)
        evaluate_official(path, metrics)
        rows.append({"repetition": repeat, "same_index_count": int(np.sum(perm == np.arange(len(perm)))), "prediction_file": path, "metrics_file": metrics})
    write_json("outputs/controls/control_manifest.json", {"seed": args.seed, "correct_metrics": "outputs/controls/correct_metrics.json", "shuffled": rows})
    test = pairing_test(correct, val, rng, args.permutations)
    write_json("outputs/controls/pairing_test.json", test)
    print("Within-subject pairing test:", test)
    print("Controls complete. Compare official identification/composite across correct vs shuffled.")

if __name__ == "__main__":
    main()

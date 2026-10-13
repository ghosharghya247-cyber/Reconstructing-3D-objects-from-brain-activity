"""Generic non-brain reference and brain-dependent coarse category retrieval."""
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.dataset import load_split
from src.descriptors import silhouette_features
from src.preprocessing import BrainPreprocessor
from src.io_utils import save_official, evaluate_official
from src.cross_validate import shape_metric_cache


def choose_medoid(features):
    mean = features.mean(axis=0, keepdims=True)
    return int(np.argmin(((features - mean) ** 2).sum(axis=1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--evaluate", action="store_true", help="Run the provided official evaluate.py")
    args = p.parse_args()
    tr = load_split(args.data_dir, "train")
    val = load_split(args.data_dir, "val")
    obj_idx = np.unique(tr["group"])
    bank_features = silhouette_features(tr["obj_masks"][obj_idx])
    bank_labels = tr["obj_class"][obj_idx]
    cd, _ = shape_metric_cache(tr["obj_points"])
    medoid_local = int(cd[np.ix_(obj_idx, obj_idx)].mean(axis=1).argmin())
    generic = np.repeat(tr["obj_points"][obj_idx[medoid_local]][None, ...], len(val["X"]), axis=0).astype(np.float32)
    generic_path = "outputs/baselines/generic_val.npz"
    save_official(generic_path, val["sample_id"], generic)
    print("Generic baseline written:", generic_path)

    pca = BrainPreprocessor(n_voxels=4096, n_components=64)
    tr_x = pca.fit_transform(tr["X"])
    vl_x = pca.transform(val["X"])
    train_labels = tr["obj_class"][tr["group"]]
    clf = LogisticRegression(C=0.1, max_iter=2000)
    clf.fit(tr_x, train_labels)
    classes = clf.predict(vl_x)
    outputs = []
    for cls in classes:
        subset = np.flatnonzero(bank_labels == cls)
        if len(subset) == 0:
            chosen = medoid_local
        else:
            chosen = int(subset[choose_medoid(bank_features[subset])])
        outputs.append(tr["obj_points"][obj_idx[chosen]])
    category = np.asarray(outputs, dtype=np.float32)
    category_path = "outputs/baselines/class_val.npz"
    save_official(category_path, val["sample_id"], category)
    print("Class baseline written:", category_path)
    if args.evaluate:
        evaluate_official(generic_path, "outputs/baselines/generic_metrics.json")
        evaluate_official(category_path, "outputs/baselines/class_metrics.json")

if __name__ == "__main__":
    main()

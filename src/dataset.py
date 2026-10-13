"""Load the NPZ layout specified in the assignment, with explicit checks."""
from pathlib import Path
import numpy as np


def load_split(data_dir, split):
    path = Path(data_dir) / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Extract the provided assignment ZIP first.")
    with np.load(path, allow_pickle=False) as f:
        data = {key: f[key] for key in f.files}
    essentials = ("X", "subject", "sample_id")
    missing = [key for key in essentials if key not in data]
    if missing:
        raise ValueError(f"{path} is missing keys: {missing}")
    n = len(data["X"])
    if data["X"].ndim != 3:
        raise ValueError(f"Expected (scans, time, voxels) X; got {data['X'].shape}")
    if len(data["subject"]) != n or len(data["sample_id"]) != n:
        raise ValueError("X / subject / sample_id lengths differ")
    if len(np.unique(data["sample_id"])) != n:
        raise ValueError("sample_id must be unique inside the split")
    if split != "test":
        need = ("group", "obj_points", "obj_masks", "obj_class")
        absent = [k for k in need if k not in data]
        if absent:
            raise ValueError(f"{path} is missing keys: {absent}")
        n_objects = len(data["obj_points"])
        if data["obj_points"].shape[1:] != (2048, 3):
            raise ValueError("obj_points must be (objects, 2048, 3)")
        if data["obj_masks"].shape[1:] != (32, 64, 64):
            raise ValueError("obj_masks must be (objects, 32, 64, 64)")
        if len(data["obj_masks"]) != n_objects:
            raise ValueError("obj_masks and obj_points object counts differ")
        if len(data["obj_class"]) != n_objects:
            raise ValueError("obj_class should be indexed by object group")
        if len(data["group"]) != n:
            raise ValueError("group length differs from scans")
        group_indices(data["group"], n_objects)
    return data


def group_indices(groups, n_objects):
    """The PDF states object arrays are indexed by group; reject invalid mapping.

    Do not assume that group labels can be matched between different NPZ files.
    """
    groups = np.asarray(groups)
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("Expected integer group indices. Inspect archive mapping before changing this.")
    groups = groups.astype(np.int64)
    if np.any(groups < 0) or np.any(groups >= n_objects):
        raise ValueError("group contains invalid obj_points indices; inspect provided NPZ conventions")
    return groups


def object_points_for_scans(data):
    return np.asarray(data["obj_points"][group_indices(data["group"], len(data["obj_points"]))], dtype=np.float32)

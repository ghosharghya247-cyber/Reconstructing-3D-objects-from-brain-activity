"""fMRI -> reduced 32-view silhouette descriptor -> nearest training 3D shape."""
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from src.dataset import group_indices
from src.preprocessing import BrainPreprocessor
from src.descriptors import ShapeDescriptor


@dataclass
class ModelConfig:
    n_voxels: int = 4096
    brain_pcs: int = 64
    shape_pcs: int = 32
    mask_side: int = 8
    ridge_alpha: float = 100.0
    feature_mode: str = "mean"
    retrieval_metric: str = "euclidean"
    seed: int = 42


class FmriShapeRetriever:
    def __init__(self, config=None):
        self.config = config or ModelConfig()

    def fit(self, X, groups, obj_masks, obj_points):
        groups = group_indices(groups, len(obj_points))
        unique, inv = np.unique(groups, return_inverse=True)
        if len(unique) < 2:
            raise ValueError("Need at least two training objects")
        self.bank_groups_ = unique
        self.bank_points_ = np.asarray(obj_points[unique], dtype=np.float32).copy()
        self.brain_ = BrainPreprocessor(
            self.config.n_voxels,
            self.config.brain_pcs,
            self.config.feature_mode,
            self.config.seed,
        )
        encoded_brain = self.brain_.fit_transform(X)
        self.shape_ = ShapeDescriptor(
            self.config.shape_pcs,
            self.config.mask_side,
            self.config.seed,
        )
        self.bank_descriptors_ = self.shape_.fit_transform(obj_masks[unique])
        labels = self.bank_descriptors_[inv]
        # Avoid giving an object more weight merely because it was scanned more often.
        counts = np.bincount(inv)
        weights = 1.0 / counts[inv]
        weights *= len(weights) / weights.sum()
        self.regressor_ = Ridge(alpha=self.config.ridge_alpha)
        self.regressor_.fit(encoded_brain, labels, sample_weight=weights)
        self.nn_ = NearestNeighbors(n_neighbors=1, metric=self.config.retrieval_metric)
        self.nn_.fit(self.bank_descriptors_)
        return self

    def predict_indices(self, X):
        descriptor_hat = self.regressor_.predict(self.brain_.transform(X))
        return self.nn_.kneighbors(descriptor_hat, return_distance=False).ravel()

    def predict(self, X):
        return self.bank_points_[self.predict_indices(X)].copy().astype(np.float32)

    def predict_train_group(self, X):
        return self.bank_groups_[self.predict_indices(X)]

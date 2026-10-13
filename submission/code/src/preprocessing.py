"""CPU-friendly fMRI features, fit only on training data."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def temporal_features(X, mode="mean"):
    if X.ndim != 3:
        raise ValueError(f"Expected 3D fMRI, got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("Non-finite raw fMRI detected; audit and clean explicitly")
    mean = np.mean(X, axis=1, dtype=np.float32)
    if mode == "mean":
        raw = mean
    elif mode == "mean_delta":
        n_time = X.shape[1]
        split = max(1, n_time // 2)
        first = np.mean(X[:, :split], axis=1, dtype=np.float32)
        second = np.mean(X[:, split:], axis=1, dtype=np.float32)
        raw = np.concatenate((mean, second - first), axis=1)
    elif mode == "delta":
        split = X.shape[1] // 2
        if split == 0:
            raise ValueError("delta requires at least two time points")
        raw = (np.mean(X[:, split:], axis=1, dtype=np.float32)
               - np.mean(X[:, :split], axis=1, dtype=np.float32))
    else:
        raise ValueError(f"Unknown temporal feature mode: {mode}")
    # Normalize within each scan; helps handle subjects unseen at inference.
    center = raw.mean(axis=1, keepdims=True)
    scale = raw.std(axis=1, keepdims=True)
    return ((raw - center) / np.maximum(scale, 1e-6)).astype(np.float32)


class BrainPreprocessor:
    def __init__(self, n_voxels=4096, n_components=64, feature_mode="mean", seed=42):
        self.n_voxels = int(n_voxels)
        self.n_components = int(n_components)
        self.feature_mode = feature_mode
        self.seed = seed

    def fit(self, X):
        f = temporal_features(X, self.feature_mode)
        var = np.var(f, axis=0)
        keep = min(max(1, self.n_voxels), f.shape[1])
        self.selected_ = np.sort(np.argsort(-var, kind="stable")[:keep])
        self.scaler_ = StandardScaler()
        z = self.scaler_.fit_transform(f[:, self.selected_])
        if self.n_components == 0:
            self.pca_ = None
            return self
        n_pca = min(self.n_components, z.shape[0] - 1, z.shape[1])
        if n_pca < 1:
            raise ValueError("Need at least two scans to fit PCA")
        self.pca_ = PCA(n_components=n_pca, svd_solver="randomized", random_state=self.seed)
        self.pca_.fit(z)
        return self

    def transform(self, X):
        f = temporal_features(X, self.feature_mode)
        z = self.scaler_.transform(f[:, self.selected_])
        return (z if self.pca_ is None else self.pca_.transform(z)).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

"""Object-level silhouette features; never fit on held-out object masks."""
import numpy as np
from sklearn.decomposition import PCA


def silhouette_features(masks, side=8):
    masks = np.asarray(masks)
    if masks.ndim != 4 or masks.shape[1:] != (32, 64, 64):
        raise ValueError(f"Expected (objects,32,64,64) silhouettes; got {masks.shape}")
    if 64 % side:
        raise ValueError("side must divide 64")
    masks = (masks > 0.5).astype(np.float32)
    block = 64 // side
    return masks.reshape(len(masks), 32, side, block, side, block).mean(axis=(3, 5)).reshape(len(masks), -1)


class ShapeDescriptor:
    def __init__(self, n_components=32, mask_side=8, seed=42):
        self.n_components = int(n_components)
        self.mask_side = int(mask_side)
        self.seed = seed

    def fit(self, masks):
        features = silhouette_features(masks, self.mask_side)
        n_pca = min(self.n_components, len(features) - 1, features.shape[1])
        if n_pca < 1:
            raise ValueError("Need at least two unique training shapes for shape PCA")
        self.pca_ = PCA(n_components=n_pca, svd_solver="randomized", random_state=self.seed)
        self.pca_.fit(features)
        return self

    def transform(self, masks):
        features = silhouette_features(masks, self.mask_side)
        return self.pca_.transform(features).astype(np.float32)

    def fit_transform(self, masks):
        return self.fit(masks).transform(masks)

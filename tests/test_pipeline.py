import numpy as np
from sklearn.model_selection import GroupKFold
from src.model import ModelConfig, FmriShapeRetriever
from src.descriptors import silhouette_features
from src.diagnostics import chamfer_fscore


def synthetic_objects(n_objects=12, scans_each=2, voxels=96):
    rng = np.random.default_rng(44)
    n = n_objects * scans_each
    groups = np.repeat(np.arange(n_objects), scans_each)
    # Synthetic masks and points are only for pipeline correctness, NOT assignment scores.
    yy, xx = np.ogrid[:64, :64]
    masks = np.zeros((n_objects, 32, 64, 64), dtype=np.uint8)
    points = np.empty((n_objects, 2048, 3), dtype=np.float32)
    for k in range(n_objects):
        rad = 8 + k
        masks[k] = (((xx - 32) ** 2 + (yy - 32) ** 2) < rad**2)
        points[k] = rng.normal(size=(2048, 3)).astype(np.float32) * (0.25 + k * 0.01)
    signature = rng.normal(size=(n_objects, voxels)).astype(np.float32)
    X = np.empty((n, 10, voxels), dtype=np.float16)
    for i, obj in enumerate(groups):
        X[i] = (signature[obj] + 0.08*rng.normal(size=(10, voxels))).astype(np.float16)
    return X, groups, masks, points


def test_descriptor_and_model_contract():
    X, groups, masks, points = synthetic_objects()
    assert silhouette_features(masks).shape == (12, 2048)
    cv = GroupKFold(3)
    tr, vl = next(cv.split(X, groups=groups))
    assert not set(groups[tr]).intersection(groups[vl])
    cfg = ModelConfig(n_voxels=64, brain_pcs=6, shape_pcs=5, ridge_alpha=2.)
    m = FmriShapeRetriever(cfg).fit(X[tr], groups[tr], masks, points)
    pred = m.predict(X[vl])
    assert pred.shape == (len(vl), 2048, 3)
    assert pred.dtype == np.float32
    assert np.isfinite(pred).all()
    assert set(m.predict_train_group(X[vl])).issubset(set(groups[tr]))
    cd, f = chamfer_fscore(pred[0], points[groups[vl[0]]])
    assert cd >= 0 and 0 <= f <= 1


def test_group_reference_validation():
    import pytest
    from src.dataset import group_indices
    assert np.array_equal(group_indices(np.array([0, 1, 1]), 2), np.array([0, 1, 1]))
    with pytest.raises(ValueError):
        group_indices(np.array([0, 3]), 2)
    with pytest.raises(ValueError):
        group_indices(np.array(["a", "b"]), 2)


def test_retrieval_points_are_unmodified_train_points():
    X, groups, masks, points = synthetic_objects()
    cfg = ModelConfig(n_voxels=48, brain_pcs=6, shape_pcs=6, ridge_alpha=10.0)
    m = FmriShapeRetriever(cfg).fit(X, groups, masks, points)
    predicted_ids = m.predict_train_group(X[:5])
    reconstructed = m.predict(X[:5])
    np.testing.assert_array_equal(reconstructed, points[predicted_ids])


def test_temporal_feature_shapes_and_values():
    from src.preprocessing import temporal_features
    X, *_ = synthetic_objects()
    for mode, ncols in (("mean", 96), ("mean_delta", 192), ("delta", 96)):
        f = temporal_features(X, mode=mode)
        assert f.shape == (len(X), ncols)
        assert f.dtype == np.float32
        assert np.isfinite(f).all()


def test_direct_delta_model_responds_to_brain_and_reloads(tmp_path):
    import joblib
    X, groups, masks, points = synthetic_objects()
    # Static offsets cancel in delta; encode the object in temporal change.
    rng = np.random.default_rng(17)
    signatures = rng.normal(size=(len(points), X.shape[-1]))
    X[:, 5:] += signatures[groups, None, :].astype(np.float16)
    cfg = ModelConfig(n_voxels=96, brain_pcs=0, shape_pcs=5,
                      feature_mode="delta", retrieval_metric="cosine", ridge_alpha=1.)
    model = FmriShapeRetriever(cfg).fit(X, groups, masks, points)
    assert model.brain_.pca_ is None
    predicted = model.predict(X)
    assert len(np.unique(model.predict_indices(X))) > 1
    np.testing.assert_array_equal(model.predict(X[::-1]), predicted[::-1])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict(X), predicted)


def test_cached_shape_scores_match_supplied_metric(tmp_path):
    from metrics import chamfer_and_fscore
    from src.cross_validate import shape_metric_cache
    _, _, _, points = synthetic_objects(n_objects=4)
    path = tmp_path / "cache.npz"
    cd, fs = shape_metric_cache(points, path)
    expected = chamfer_and_fscore(points[1], points[3])
    np.testing.assert_allclose([cd[1, 3], fs[1, 3]], expected)
    again = shape_metric_cache(points, path)
    np.testing.assert_array_equal(again[0], cd)
    changed = points.copy()
    changed[0] += 1
    rebuilt, _ = shape_metric_cache(changed, path)
    assert rebuilt[0, 1] != cd[0, 1]


def test_cluster_bootstrap_keeps_object_scans_together():
    from src.uncertainty import cluster_bootstrap
    # A single object is one resampling unit even with different scan scores.
    scores = np.array([0., 1., 2.])
    result = cluster_bootstrap(scores, np.zeros(3), np.random.default_rng(7), 100)
    assert result['mean'] == result['ci95_lower'] == result['ci95_upper'] == 1.


def test_pairing_statistic_matches_official_identification():
    from src.controls import pairing_test
    from metrics import chamfer_and_fscore
    _, _, _, points = synthetic_objects(n_objects=4)
    groups = np.repeat(np.arange(4), 2)
    predicted = points[groups]
    val = {'group': groups, 'subject': np.array(['a','b']*4), 'obj_points': points}
    result = pairing_test(predicted, val, np.random.default_rng(8), 200)
    assert result['observed_identification'] == 100.
    assert result['p_value'] < .05

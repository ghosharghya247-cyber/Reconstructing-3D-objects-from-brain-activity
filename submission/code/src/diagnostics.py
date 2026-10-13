"""Diagnostic geometry metrics, not a replacement for supplied metrics.py."""
import numpy as np
from scipy.spatial import cKDTree


def chamfer_fscore(pred, target, threshold=0.05):
    a, b = np.asarray(pred, dtype=np.float32), np.asarray(target, dtype=np.float32)
    da = cKDTree(b).query(a, k=1)[0]
    db = cKDTree(a).query(b, k=1)[0]
    chamfer = float(da.mean() + db.mean())
    precision = float((da < threshold).mean())
    recall = float((db < threshold).mean())
    fscore = 2 * precision * recall / (precision + recall + 1e-12)
    return chamfer, float(fscore)


def scan_diagnostics(pred, targets):
    rows = [chamfer_fscore(p, y) for p, y in zip(pred, targets)]
    return {
        "mean_diagnostic_chamfer": float(np.mean([r[0] for r in rows])),
        "mean_diagnostic_fscore": float(np.mean([r[1] for r in rows])),
    }

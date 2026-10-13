# Method and limitations

The real-data audit and execution guide are in [steps.md](steps.md). Source scripts are our implementation; the supplied scoring, export and checking helpers were retained unchanged.

- Training object groups remain together in every CV fold; transforms and retrieval banks are fitted within folds.
- Group IDs are local indices within each split and must not be equated across files.
- The selected model uses temporal delta features, no brain PCA, Ridge descriptor prediction and cosine training-shape retrieval. It receives only fMRI at inference and uses no pretrained models.
- Retrieval preserves canonical coordinates but cannot generate novel geometry.
- Bootstrap intervals resample whole object groups. They condition on fixed validation opponents, fixed composite anchors and the trained model.
- Global shuffled controls can mix subjects; a separate within-subject pairing test is reported. Neither implements the organisers' unpublished hidden significance gate.
- Validation identification improved to 60.30%, but geometry is modest and composite uncertainty includes zero. Hidden-test performance remains unknown.
- The exact generic baseline is the supplied mean-Chamfer training medoid, using a target-only scoring cache; that cache never enters fitting.
- The report is generated from measured JSON results by `src.report`, is three pages, and contains both required dataset citations.
- The upstream licence file referenced by the PDF was absent from this workspace. Preserve it from the original archive if available.

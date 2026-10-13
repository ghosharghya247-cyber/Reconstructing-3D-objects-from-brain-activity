# fMRI to 3D reconstruction

CPU model and complete submission workflow for `fmri_3d_reconstruction_assignment.pdf`. The provided dataset and original scoring/export helpers are already present in this workspace.

Read **[steps.md](steps.md)** for the assignment audit, the data and model explanation, every input/output location, stage commands, measured results, runtime and troubleshooting.

```bash
cd /home/arghya/fmri-3d-reconstruction
source .venv/bin/activate
python run_project.py --check-only    # inspect existing submission
python run_project.py --reuse-config # rebuild with selected config
python run_project.py                # grouped selection + complete rebuild
```

For a fresh environment, use Python 3.12 and `python -m pip install -r requirements.lock.txt`. The smaller `requirements.txt` specifies compatible version ranges. No GPU or pretrained models are required.

The model predicts training silhouette descriptors from temporal fMRI changes using Ridge regression, then retrieves a canonical training point cloud. The workflow performs grouped cross-validation, official validation scoring, object-clustered uncertainty, shuffled-brain controls, test inference, PLY/render export, a three-page report, source packaging, integrity checks and `submission.zip` creation.

The submission checker validates file completeness. The organisers separately assess brain dependence and hidden-test scores. Validation evidence and limitations are documented in `steps.md` and `submission/report.pdf`.

When running the packaged source, work in `submission/code/` and place the provided dataset in `submission/code/data/`. Its checkpoint is `artifacts/best.joblib`. Data and exploratory predictions are excluded from the submission archive.

Dataset citation: Gao et al., MinD-3D (arXiv:2312.07485) and MinD-3D++ (arXiv:2409.11315). The assignment identifies the dataset licence as Apache-2.0; the referenced `LICENSE_fMRI-Shape.txt` was absent from this workspace.

"""Build a three-page PDF from measured results, never placeholders."""
import argparse
import json
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from src.dataset import load_split
from starter import load_cameras, render_points


def read(path):
    return json.loads(Path(path).read_text())


def page(title):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(.07, .95, title, fontsize=17, weight='bold', va='top')
    return fig


def paragraph(fig, y, title, body, width=105, size=10):
    fig.text(.07, y, title, fontsize=11, weight='bold', va='top')
    lines = '\n'.join(textwrap.wrap(body, width=width))
    fig.text(.07, y-.025, lines, fontsize=size, va='top', linespacing=1.45)
    return y - .025 - .018 * (size / 10) * (len(lines.splitlines()) + 1)


def table(fig, box, rows, columns, size=9):
    ax = fig.add_axes(box)
    ax.axis('off')
    tb = ax.table(cellText=rows, colLabels=columns, loc='upper center', cellLoc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(size)
    tb.scale(1, 1.65)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='submission/report.pdf')
    args = p.parse_args()
    config = read('artifacts/train_info.json')['config']
    cv = read('outputs/validation/cv_results.json')
    final = read('submission/metrics/val_metrics.json')
    unc = read('outputs/validation/official_uncertainty.json')
    pairing = read('outputs/controls/pairing_test.json')
    runtime = read('outputs/audit/runtime.json')
    variants = [('Generic medoid', read('outputs/baselines/generic_metrics.json')),
                ('Category retrieval', read('outputs/baselines/class_metrics.json')),
                ('Selected model', final)]
    shuffled = [read(f'outputs/controls/shuffled_{i:02d}_metrics.json') for i in range(3)]
    keys = ['chamfer', 'fscore', 'silhouette_iou', 'silhouette_ssim', 'identification', 'composite']
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        fig = page('Reconstructing 3D objects from fMRI')
        y = paragraph(fig, .9, 'Data and inference contract',
            'Provided data only: 533 training scans / 292 objects, 125 validation scans / 67 objects, '
            '167 blind test scans. Each scan has 10 time points and 31,141 voxels. Predictions contain '
            '2,048 surface points in the supplied canonical frame (origin centred, z up). '
            'Test inference reads fMRI only; subject IDs are available but unused by the selected model. '
            'No pretrained models or external object models are used.')
        y = paragraph(fig, y-.02, 'Approach',
            f'Temporal features: {config["feature_mode"]}. delta means the last-five-time-point mean minus the first-five mean; '
            'mean_delta concatenates temporal mean and delta. Spatial standardization is performed separately '
            f'for each scan. Keep {config["n_voxels"]} highest-variance features, then train-fitted voxel scaling. '
            f'Brain PCA components: {config["brain_pcs"]} (0 means no PCA). '
            'For each unique training object, downsample all 32 binary silhouettes from 64 x 64 to '
            f'{config["mask_side"]} x {config["mask_side"]}, then fit {config["shape_pcs"]} shape PCs. '
            f'Weighted Ridge (alpha {config["ridge_alpha"]:g}) predicts shape descriptors; each object has equal total regression weight. '
            f'Retrieve the nearest training descriptor using {config.get("retrieval_metric", "euclidean")} distance and copy its 3D points unchanged. '
            'Training images and voxel-map coordinates are not used.')
        y = paragraph(fig, y-.02, 'Selection and leakage prevention',
            'Five GroupKFold splits on training object IDs. Feature selection, scaling, brain PCA, shape PCA, '
            'Ridge and retrieval bank are fitted using each training fold only. A target-only pairwise shape '
            'cache accelerates scoring and never enters fitting. Select highest grouped-CV identification, '
            'with Chamfer as tie-breaker. Candidate additions were motivated by three-fold training category '
            'experiments; validation is reported after grouped selection. No hidden test targets are available.')
        rows = [[r['candidate'], r['config']['feature_mode'], r['config']['brain_pcs'], f"{r['config']['ridge_alpha']:g}", r['config'].get('retrieval_metric','euclidean'),
                 f"{r['mean_diagnostic_chamfer']:.4f}", f"{r['mean_identification']:.2f}%"] for r in cv['results']]
        table(fig, [.07, .13, .86, .22], rows, ['Candidate','Features','Brain PCs','Alpha','Retrieval','CV Chamfer','CV ident.'], 8)
        fig.text(.07, .085, f"Selected candidate {cv['best_candidate']}; seed {config['seed']}. CV identification uses only fold opponent objects.", fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        fig = page('Validation results, uncertainty and controls')
        table(fig, [.07, .73, .86, .17], [[name]+[f'{s[k]:.4f}' for k in keys] for name,s in variants],
              ['Method','Chamfer','F@0.05','IoU','SSIM','Ident. %','Composite'], 8)
        fig.text(.07,.715,'Metrics from supplied evaluate.py; Chamfer lower is better, other metrics higher. Composite is unclipped.',fontsize=9)
        rows = [[k, f"{unc['metrics'][k]['mean']:.4f}",
                 f"[{unc['metrics'][k]['ci95_lower']:.4f}, {unc['metrics'][k]['ci95_upper']:.4f}]"] for k in keys]
        table(fig, [.1,.46,.8,.23], rows, ['Metric','Mean','95% group-bootstrap interval'], 9)
        y=paragraph(fig,.445,'Uncertainty scope',
            f"{unc['iterations']} percentile-bootstrap replicates resample 67 whole object groups, retaining all scans "
            'from each selected group. Identification retains the original opponent set; composite uses fixed '
            "generic/oracle anchors. These are conditional intervals, not refitted-model intervals or the hidden gate's test.")
        table(fig,[.15,.23,.7,.11], [[f'Shuffle {i}',f"{s['identification']:.2f}%",f"{s['composite']:+.4f}"] for i,s in enumerate(shuffled)],
              ['Global fMRI pairing control','Identification','Composite'],9)
        status = 'supported by these validation checks' if (final['identification']>50 and pairing['p_value']<.05 and unc['metrics']['identification']['ci95_lower']>50) else 'not established by these validation checks'
        paragraph(fig,.21,'Brain-use evidence',
            f"Correct pairing identification: {final['identification']:.2f}%. Within-subject pairing permutation "
            f"test ({pairing['iterations']} permutations): null mean {pairing['null_mean_identification']:.2f}%, "
            f"one-sided p={pairing['p_value']:.4g}. Brain dependence is {status}. The supplied evaluator and checker "
            'do not implement the hidden significance gate; hidden-test passage cannot be guaranteed.')
        pdf.savefig(fig);plt.close(fig)

        fig=page('Examples, limitations and reproducibility')
        val=load_split('data','val')
        with np.load('outputs/val_predictions.npz') as z: pred=z['points']
        per=read('outputs/validation/per_scan_metrics.json')['metrics']
        order=np.argsort([r['chamfer'] for r in per]);cams=load_cameras()
        for row, ix in enumerate([int(order[0]),int(order[-1])]):
            for col, points in enumerate([val['obj_points'][val['group'][ix]],pred[ix]]):
                ax=fig.add_axes([.12+col*.42,.71-row*.22,.32,.17])
                ax.imshow(render_points(points,cams,0),cmap='gray',vmin=0,vmax=255);ax.axis('off')
                ax.set_title(('Ground truth' if col==0 else 'Prediction')+f' / {val["sample_id"][ix]}',fontsize=9)
            fig.text(.12,.69-row*.22,f"{'Best' if row==0 else 'Worst'} validation Chamfer: {per[ix]['chamfer']:.4f}; examples selected by observed error.",fontsize=9)
        y=paragraph(fig,.43,'Limitations and what did / did not work',
            'Retrieval preserves realistic canonical geometry but cannot generate new shapes or concavities. '
            'The original high-variance PCA baseline had 49.43% validation identification and composite -0.0396. '
            'Training category experiments favored temporal delta features; grouped CV determined the final choice. '
            'A constant training medoid can score reasonably on geometry while carrying no scan information. '
            'Subjects have unequal sample sizes. Global shuffled controls may mix subjects; the separate pairing '
            'test preserves subjects. Geometry remains weaker than the medoid; the composite interval spans zero.',width=115,size=9)
        measured=sum(r['seconds'] for r in runtime['stages'])
        y=paragraph(fig,y-.015,'Reproduction and runtime',
            f"Python {runtime['python']}, CPU only, {runtime['threads']} BLAS threads; {runtime['logical_cpus']} logical CPUs available. "
            f"Recorded completed stage time: {measured:.1f} seconds (CV {'reused' if runtime['reuse_config'] else 'included'}). "
            'Run: python run_project.py; to reuse the selected config: python run_project.py --reuse-config. '
            'See code/steps.md for environment setup, stage commands, expected runtime, outputs and checks. '
            'Exact dependency versions: code/requirements.lock.txt; packaged timings: code/evidence/runtime.json. '
            'The supplied data must be placed in code/data when running the packaged source.',width=115,size=9)
        paragraph(fig,y-.01,'Citation and licence',
            'Gao et al., MinD-3D (arXiv:2312.07485) and MinD-3D++ (arXiv:2409.11315). '
            'The PDF identifies fMRI-Shape as Apache-2.0. LICENSE_fMRI-Shape.txt was not present in this workspace; '
            'retain the original licence from the provided archive if available.',width=108,size=9)
        pdf.savefig(fig);plt.close(fig)
    print(f'Saved {args.out} (3 pages)')


if __name__ == '__main__':
    main()

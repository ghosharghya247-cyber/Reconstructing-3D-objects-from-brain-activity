"""Run the assignment workflow from this file's directory; CPU only."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reuse-config', action='store_true', help='Skip grouped CV and use configs/best.yaml')
    p.add_argument('--check-only', action='store_true', help='Check existing outputs without retraining')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.threads < 1:
        p.error('--threads must be positive')
    root = Path(__file__).resolve().parent
    os.chdir(root)
    env = os.environ.copy()
    env.update({key: str(args.threads) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
    for name in ('train', 'val', 'test', 'cameras', 'voxels'):
        if not (root / 'data' / f'{name}.npz').is_file():
            p.error(f'Missing data/{name}.npz: place the provided assignment data beside this script (see steps.md)')
    records = []
    logdir = root / 'outputs' / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    runtime_file = root / 'outputs' / 'audit' / ('check_runtime.json' if args.check_only else 'runtime.json')
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'cpu': platform.processor(), 'logical_cpus': os.cpu_count(), 'threads': args.threads,
        'reuse_config': args.reuse_config,
        'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'matplotlib', 'Pillow', 'joblib', 'PyYAML')},
        'stages': records,
    }

    def stage(name, *arguments):
        cmd = [sys.executable, *arguments]
        print(f'[{name}] {" ".join(cmd)}', flush=True)
        started = time.perf_counter()
        with (logdir / f'{name}.log').open('w') as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        records.append({'stage': name, 'command': cmd, 'seconds': time.perf_counter() - started, 'exit_code': result.returncode})
        runtime_file.write_text(json.dumps(metadata, indent=2))
        print((logdir / f'{name}.log').read_text(), flush=True)
        if result.returncode:
            raise SystemExit(f'{name} failed. See outputs/logs/{name}.log; later stages were not run.')

    stage('tests', '-m', 'pytest', '-q')
    stage('audit', '-m', 'src.audit')
    if not args.check_only:
        if not args.reuse_config:
            stage('cv', '-m', 'src.cross_validate', '--folds', '5')
        stage('train', '-m', 'src.train')
        stage('baselines', '-m', 'src.baselines', '--evaluate')
        stage('validation_prediction', '-m', 'src.predict', '--split', 'val', '--output', 'outputs/val_predictions.npz')
        stage('validation_metrics', 'evaluate.py', 'outputs/val_predictions.npz', '--save', 'submission/metrics/val_metrics.json')
        stage('uncertainty', '-m', 'src.uncertainty')
        stage('controls', '-m', 'src.controls', '--repeats', '3')
        stage('test_prediction', '-m', 'src.predict', '--split', 'test', '--output', 'submission/predictions.npz')
        stage('export', '-m', 'src.export')
        stage('report', '-m', 'src.report')
        stage('package', '-m', 'src.package')
    stage('integrity', '-m', 'src.verify')
    stage('official_check', 'check_submission.py', 'submission/')
    if not args.check_only:
        with zipfile.ZipFile('submission.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(Path('submission').rglob('*')):
                if file.is_file() and '__pycache__' not in file.parts and file.suffix != '.pyc':
                    archive.write(file, file.as_posix())
        print('Created submission.zip', flush=True)
    print(f'Completed in {sum(r["seconds"] for r in records):.1f} seconds. Timing details: {runtime_file.relative_to(root)}')


if __name__ == '__main__':
    main()

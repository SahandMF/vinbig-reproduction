#!/usr/bin/env python
import argparse
import os
import sys
import subprocess


def main():
    # Parse CV-specific args and forward the rest to train.py
    parser = argparse.ArgumentParser(description='EfficientDet CV Orchestrator')
    parser.add_argument('root', metavar='DIR', help='path to dataset root (same as train.py)')
    parser.add_argument('--cv', type=int, default=5, help='Number of folds (default: 5)')
    parser.add_argument('--cv-dir', type=str, required=True,
                        help='Directory containing train_fold_{k}.json and val_fold_{k}.json')
    parser.add_argument('--output-base', type=str, default='',
                        help='Base output directory; each fold will append fold_{k}')
    # Accept all other args and pass them through to train.py
    args, passthrough = parser.parse_known_args()

    train_py = os.path.join(os.path.dirname(__file__), 'train.py')

    for k in range(args.cv):
        ann_train = os.path.join(args.cv_dir, f'train_fold_{k}.json')
        ann_val = os.path.join(args.cv_dir, f'val_fold_{k}.json')
        if not (os.path.exists(ann_train) and os.path.exists(ann_val)):
            print(f'[CV] Skipping fold {k}: missing files')
            continue

        out_dir = args.output_base or os.path.join(os.path.dirname(__file__), 'output', 'cv_runs')
        out_dir = os.path.join(out_dir, f'fold_{k}')
        os.makedirs(out_dir, exist_ok=True)

        cmd = [sys.executable, train_py, args.root]
        cmd += passthrough
        cmd += [
            '--ann-train-file', ann_train,
            '--ann-val-file', ann_val,
            '--output', out_dir,
        ]

        print(f'\n[CV] Running fold {k}:')
        print(' ', ' '.join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f'[CV] Fold {k} failed with code {result.returncode}. Aborting.')
            sys.exit(result.returncode)

    print('\n[CV] All folds completed.')


if __name__ == '__main__':
    main()



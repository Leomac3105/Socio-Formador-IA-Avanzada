#!/usr/bin/env python3
"""Filter VLM labels by confidence and create train/val splits.

Reads `Playground/data/graph/with_objects/vlm_labels_clip.csv` with columns
`file,label,conf_weight` and filters rows with `conf_weight >= --min-conf`.
Saves filtered CSV, `train.csv` and `val.csv` (stratified split).

Usage:
  python Playground/scripts/filter_and_split_labels.py --min-conf 0.6 --out-dir Playground/data/graph/with_objects
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels-csv', type=Path, default=Path('Playground/data/graph/with_objects/vlm_labels_clip.csv'))
    parser.add_argument('--min-conf', type=float, default=0.6)
    parser.add_argument('--out-dir', type=Path, default=Path('Playground/data/graph/with_objects'))
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--min-samples-per-class', type=int, default=5, help='Drop classes with fewer samples after filtering')
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    print('Total labeled rows:', len(df))
    print('Counts per label (all):')
    print(df['label'].value_counts())

    df['conf_weight'] = df['conf_weight'].astype(float)
    df_f = df[df['conf_weight'] >= args.min_conf].copy()
    print(f'After filtering conf >= {args.min_conf}: {len(df_f)} rows')
    print('Counts per label (filtered):')
    print(df_f['label'].value_counts())

    # Drop classes with too few samples
    counts = df_f['label'].value_counts()
    drop = counts[counts < args.min_samples_per_class].index.tolist()
    if len(drop) > 0:
        print('Dropping classes with fewer than', args.min_samples_per_class, 'samples:', drop)
        df_f = df_f[~df_f['label'].isin(drop)]

    filtered_csv = out_dir / f'vlm_labels_filtered_{args.min_conf:.2f}.csv'
    df_f.to_csv(filtered_csv, index=False)
    print('Saved filtered labels to', filtered_csv)

    if len(df_f) == 0:
        print('No rows after filtering. Exiting.')
        return

    # Stratified split
    train, val = train_test_split(df_f, test_size=args.test_size, stratify=df_f['label'], random_state=args.random_state)
    train_csv = out_dir / 'train.csv'
    val_csv = out_dir / 'val.csv'
    train.to_csv(train_csv, index=False)
    val.to_csv(val_csv, index=False)
    print('Saved train/val:', train_csv, val_csv)

    # Save low-confidence list for review
    low_conf = df[df['conf_weight'] < args.min_conf]
    low_csv = out_dir / f'vlm_labels_lowconf_below_{args.min_conf:.2f}.csv'
    low_conf.to_csv(low_csv, index=False)
    print('Saved low-confidence rows to', low_csv)

    # Summary
    print('\nTrain counts per label:')
    print(train['label'].value_counts())
    print('\nVal counts per label:')
    print(val['label'].value_counts())


if __name__ == '__main__':
    main()

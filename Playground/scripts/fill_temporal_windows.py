#!/usr/bin/env python3
"""Fill temporal gaps in .npy skeleton windows.

For each window file (shape (T,K,17,2)) this script interpolates missing
joint coordinates over time per-person/per-joint. Missing values are assumed
to be exact zeros (both x and y == 0). The default behaviour is linear
interpolation across valid frames and forward/back-fill for edges.

Outputs are saved to `Playground/data/npy_filled` mirroring input filenames.

Usage:
  python Playground/scripts/fill_temporal_windows.py --in-dir Playground/data/npy --out-dir Playground/data/npy_filled
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


def fill_window_linear(window: np.ndarray) -> np.ndarray:
    # window: (T, K, V, 2)
    T, K, V, D = window.shape
    out = window.astype(np.float32).copy()

    # For each person and joint, interpolate x and y separately
    times = np.arange(T)
    for m in range(K):
        for v in range(V):
            x = out[:, m, v, 0]
            y = out[:, m, v, 1]
            # detect valid if not both zero
            valid_x = ~(np.isclose(x, 0.0) & np.isclose(y, 0.0))
            if valid_x.sum() == 0:
                # nothing to do
                continue
            # set invalid to nan for interpolation
            x_nan = x.copy(); y_nan = y.copy()
            x_nan[~valid_x] = np.nan
            y_nan[~valid_x] = np.nan

            # Interpolate using numpy.interp where possible
            # handle x
            xv = x_nan
            if np.isnan(xv).any():
                valid_idx = np.where(~np.isnan(xv))[0]
                if valid_idx.size > 0:
                    # linear interp for interior
                    interp_x = np.interp(times, valid_idx, xv[valid_idx])
                    # where valid, keep original to avoid numeric diff
                    xv = np.where(np.isnan(x_nan), interp_x, x_nan)
                else:
                    xv = x
            # handle y
            yv = y_nan
            if np.isnan(yv).any():
                valid_idx = np.where(~np.isnan(yv))[0]
                if valid_idx.size > 0:
                    interp_y = np.interp(times, valid_idx, yv[valid_idx])
                    yv = np.where(np.isnan(y_nan), interp_y, y_nan)
                else:
                    yv = y

            # For any leading/trailing NaNs (np.interp filled using edge values), np.interp already forward/backfilled
            out[:, m, v, 0] = xv
            out[:, m, v, 1] = yv

    return out


def process_all(in_dir: Path, out_dir: Path, inplace: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob('*_win*.npy'))
    for f in tqdm(files, desc='Filling windows'):
        try:
            arr = np.load(f)
        except Exception as e:
            print('Failed to load', f, e)
            continue
        if arr.ndim == 5:
            # (N, T, K, V, 2) - process each window individually
            out_list = []
            for i in range(arr.shape[0]):
                win = arr[i]
                filled = fill_window_linear(win)
                out_list.append(filled)
            out_arr = np.stack(out_list, axis=0)
        elif arr.ndim == 4:
            out_arr = fill_window_linear(arr)
        else:
            print('Skipping unsupported shape', f, arr.shape)
            continue

        if inplace:
            np.save(f, out_arr)
        else:
            out_path = out_dir / f.name
            np.save(out_path, out_arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', type=Path, default=Path('Playground/data/npy'))
    parser.add_argument('--out-dir', type=Path, default=Path('Playground/data/npy_filled'))
    parser.add_argument('--inplace', action='store_true', help='Overwrite input files')
    args = parser.parse_args()
    process_all(args.in_dir, args.out_dir, inplace=args.inplace)


if __name__ == '__main__':
    main()

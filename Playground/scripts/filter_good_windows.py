#!/usr/bin/env python3
"""Filter good windows for VLM labeling and render representative frames for them.

Checks each `.npy` in `npy_filled` for minimum number of non-empty frames and
at least one person present. Creates:
- `good_windows.txt` (full paths to good .npy)
- `bad_windows.txt` (full paths to rejected .npy)
- `candidate_frames.txt` (npy_path \t frame_image_path) for the good windows

Also renders a single frame per good window (first non-empty frame) into the
frames output directory.

Usage:
  python Playground/scripts/filter_good_windows.py --in-dir Playground/data/npy_filled --out-frames Playground/data/graph/with_objects/frames_filled --min-frames 3
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from collections import Counter

# skeleton edges (COCO-like 17) - reuse from render script
COCO_EDGES = [
    (0,1),(1,3),(0,2),(2,4),(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),(5,6),(11,12),(1,2),(0,5),(0,6)
]


def is_frame_nonempty(frame: np.ndarray) -> bool:
    return not np.allclose(frame, 0)


def first_nonempty_frame(window: np.ndarray):
    T = window.shape[0]
    for t in range(T):
        if is_frame_nonempty(window[t]):
            return t
    return None


def render_frame(window: np.ndarray, frame_idx: int, out_path: Path, image_size=256):
    frame = window[frame_idx]
    img = Image.new('RGB', (image_size, image_size), (255,255,255))
    draw = ImageDraw.Draw(img)
    T,K,V,_ = window.shape
    for m in range(K):
        joints = frame[m]
        pts = [(float(x)*(image_size-1), float(y)*(image_size-1)) for x,y in joints]
        for a,b in COCO_EDGES:
            if a < V and b < V:
                xa,ya = pts[a]; xb,yb = pts[b]
                draw.line((xa,ya,xb,yb), fill=(30,30,160), width=2)
        for (x,y) in pts:
            r=3
            draw.ellipse((x-r,y-r,x+r,y+r), fill=(200,30,30))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', type=Path, default=Path('Playground/data/npy_filled'))
    parser.add_argument('--out-frames', type=Path, default=Path('Playground/data/graph/with_objects/frames_filled'))
    parser.add_argument('--min-frames', type=int, default=3, help='Minimum non-empty frames required to consider a window good')
    parser.add_argument('--min-persons', type=int, default=1, help='Minimum persons with any data')
    parser.add_argument('--out-dir', type=Path, default=Path('Playground/data/graph/with_objects'))
    args = parser.parse_args()

    in_dir = args.in_dir
    out_frames = args.out_frames
    out_dir = args.out_dir
    out_frames.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(in_dir.glob('*_win*.npy')))
    good = []
    bad = []
    for f in files:
        try:
            arr = np.load(f)
        except Exception as e:
            print('skip load error', f, e)
            bad.append(f)
            continue
        if arr.ndim == 5:
            window = arr[0]
        else:
            window = arr
        T,K,V,_ = window.shape
        nonempty_count = sum(1 for t in range(T) if is_frame_nonempty(window[t]))
        persons_nonzero = sum(1 for m in range(K) if not np.allclose(window[:,m,:,:], 0))
        if nonempty_count >= args.min_frames and persons_nonzero >= args.min_persons:
            good.append((f, nonempty_count, persons_nonzero))
        else:
            bad.append((f, nonempty_count, persons_nonzero))

    good_txt = out_dir / 'good_windows.txt'
    bad_txt = out_dir / 'bad_windows.txt'
    candidate_txt = out_frames.parent / 'candidate_frames.txt'

    with good_txt.open('w') as fg:
        for f,fn,pn in good:
            fg.write(f'{f}\t{fn}\t{pn}\n')
    with bad_txt.open('w') as fb:
        for f,fn,pn in bad:
            fb.write(f'{f}\t{fn}\t{pn}\n')

    # Render frames only for good windows and produce candidate_frames.txt
    with candidate_txt.open('w') as fc:
        for f,fn,pn in good:
            arr = np.load(f)
            window = arr[0] if arr.ndim==5 else arr
            idx = first_nonempty_frame(window)
            if idx is None:
                print('no non-empty frame for', f)
                continue
            out_img = out_frames / (f.stem + '.png')
            render_frame(window, idx, out_img)
            fc.write(f'{f}\t{out_img}\n')

    print('Good windows:', len(good), 'Bad windows:', len(bad))
    print('Wrote:', good_txt, bad_txt, candidate_txt)


if __name__ == '__main__':
    main()

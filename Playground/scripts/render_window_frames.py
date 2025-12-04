#!/usr/bin/env python3
"""Renderiza un fotograma representativo (imagen) para cada ventana `.npy`.

Lee ventanas en `Playground/data/npy` (por defecto), toma el frame central
de cada ventana y dibuja los esqueletos (joints y conexiones) en una imagen PNG.
Genera `candidate_frames.txt` con pares `npy_path\timage_path` listo para etiquetar.

Uso:
  python Playground/scripts/render_window_frames.py --in-dir Playground/data/npy --out-dir Playground/data/graph/with_objects/frames
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# COCO 17 skeleton edges (0-based)
COCO_EDGES = [
    (0,1),(1,3),(0,2),(2,4),(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),(5,6),(11,12),(1,2),(0,5),(0,6)
]


def ensure_window(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 5:
        return arr[0]
    if arr.ndim == 4:
        return arr
    raise ValueError('Input .npy debe tener shape (T,K,17,2) o (N,T,K,17,2)')


def render_window(window: np.ndarray, out_path: Path, image_size=256):
    # window: (T, K, V=17, 2)
    T, K, V, _ = window.shape

    def _count_visible_joints(frame: np.ndarray) -> int:
        # frame: (K, V, 2) -> count joints visible in at least one person
        vis = 0
        for v in range(frame.shape[1]):
            if np.any(np.linalg.norm(frame[:, v, :], axis=1) > 0):
                vis += 1
        return vis

    # Choose frame: prefer the one with the most visible joints (heuristic)
    best_idx = 0
    best_vis = -1
    for t in range(T):
        vis = _count_visible_joints(window[t])
        if vis > best_vis:
            best_vis = vis
            best_idx = t
    frame_idx = best_idx
    frame = window[frame_idx]

    img = Image.new('RGB', (image_size, image_size), (255,255,255))
    draw = ImageDraw.Draw(img)

    # Draw connections
    for m in range(K):
        joints = frame[m]  # (V,2) normalized (0..1 assumed)
        # scale to pixels
        pts = [(float(x) * (image_size-1), float(y) * (image_size-1)) for x,y in joints]
        # draw edges
        for a,b in COCO_EDGES:
            if a < V and b < V:
                xa, ya = pts[a]
                xb, yb = pts[b]
                draw.line((xa, ya, xb, yb), fill=(30,30,160), width=2)
        # draw joints
        for (x,y) in pts:
            r = 3
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(200,30,30))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return best_vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', type=Path, default=Path('Playground/data/npy'))
    parser.add_argument('--out-dir', type=Path, default=Path('Playground/data/graph/with_objects/frames'))
    parser.add_argument('--ext', type=str, default='.npy')
    parser.add_argument('--min-joints', type=int, default=12, help='mínimo de joints visibles (de 17) para incluir la ventana')
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_txt = out_dir.parent / 'candidate_frames.txt'
    skipped_txt = out_dir.parent / 'candidate_frames_skipped.txt'
    with candidate_txt.open('w') as ftxt, skipped_txt.open('w') as fskip:
        for npy in sorted(in_dir.glob(f'*{args.ext}')):
            try:
                arr = np.load(npy)
                window = ensure_window(arr)
            except Exception as e:
                print('Skipping', npy, 'error loading:', e)
                continue
            img_name = npy.stem + '.png'
            out_img = out_dir / img_name
            try:
                vis = render_window(window, out_img)
                if vis >= args.min_joints:
                    ftxt.write(f"{npy}\t{out_img}\t{vis}\n")
                else:
                    fskip.write(f"{npy}\t{out_img}\t{vis}\n")
            except Exception as e:
                print('Failed rendering', npy, e)
    print('Wrote candidate list to', candidate_txt)
    print('Wrote skipped list to', skipped_txt)


if __name__ == '__main__':
    main()

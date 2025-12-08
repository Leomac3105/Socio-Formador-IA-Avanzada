#!/usr/bin/env python3
"""Quick smoke-test: run MP-GCN forward pass on saved panoramic graph files.

Loads files produced by `build_panoramic_graph.py` (``*_X.npy``, ``*_A0.npy``,
``*_A_intra.npy``, ``*_A_inter.npy``), builds a minimal adjacency tensor and
runs a single forward pass on CPU (or CUDA if available).

Usage: python Playground/scripts/run_forward_on_graphs.py --graph-dir data/graph/with_objects --max-samples 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch


def normalize_digraph(A: np.ndarray) -> np.ndarray:
    Dl = A.sum(0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node), dtype=np.float32)
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = A.dot(Dn)
    return AD


def build_A_stack(A0: np.ndarray, A_intra: np.ndarray, A_inter: np.ndarray, required_slices: int = 3) -> np.ndarray:
    # Normalize each adjacency slice similarly to Graph._normalize_digraph
    a0 = normalize_digraph(A0)
    ai = normalize_digraph(A_intra)
    ae = normalize_digraph(A_inter)
    stack = [a0, ai, ae]
    # If more slices are required, repeat the last one
    while len(stack) < required_slices:
        stack.append(stack[-1])
    return np.stack(stack[:required_slices]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-dir', type=Path, default=Path('Playground/data/graph/with_objects'))
    parser.add_argument('--max-samples', type=int, default=10)
    parser.add_argument('--camera-filter', type=str, default='columpioscam3')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-class', type=int, default=8)
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    if not graph_dir.exists():
        print(f'Graph dir not found: {graph_dir}')
        sys.exit(1)

    # Add repo root to sys.path so we can import MPGCN
    # script path: .../Socio-Formador-IA-Avanzada/Playground/scripts/...
    # parents[2] -> repo root (Socio-Formador-IA-Avanzada)
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # Also add the MP-GCN subproject path if present (it contains a top-level `src` folder)
    mpgcn_root = repo_root / '01-replica-paper' / 'MP-GCN'
    if mpgcn_root.exists() and str(mpgcn_root) not in sys.path:
        sys.path.insert(0, str(mpgcn_root))

    try:
        from src.model.MPGCN.nets import MPGCN
    except Exception as e:
        print('Error importing MPGCN:', e)
        sys.exit(1)

    files = sorted(graph_dir.glob(f'*{args.camera_filter}*_X.npy'))
    if len(files) == 0:
        # fallback: any _X.npy
        files = sorted(graph_dir.glob('*_X.npy'))

    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print('Using device:', device)

    results = []
    done = 0
    for x_path in files:
        if done >= args.max_samples:
            break
        stem = x_path.stem.replace('_X', '')
        a0_path = graph_dir / f'{stem}_A0.npy'
        ai_path = graph_dir / f'{stem}_A_intra.npy'
        ae_path = graph_dir / f'{stem}_A_inter.npy'
        if not (a0_path.exists() and ai_path.exists() and ae_path.exists()):
            print(f'Skipping (missing A files): {x_path.name}')
            continue

        X = np.load(x_path)  # (C, T, Vp, M)
        A0 = np.load(a0_path)
        A_intra = np.load(ai_path)
        A_inter = np.load(ae_path)

        C, T, Vp, M = X.shape
        # Build adjacency stack with required number of slices (kernel distance +1)
        required = 3
        A_stack = build_A_stack(A0, A_intra, A_inter, required_slices=required)

        # Prepare model input: (N, I, C, T, V, M) with I=1, N=1
        data = torch.from_numpy(X).float().unsqueeze(0).unsqueeze(0).to(device)
        data_shape = (1, C, T, Vp, M)

        # Prepare simple parts grouping: single part with all joints
        parts = [np.arange(Vp)]

        model_kwargs = {
            'use_att': True,
            'reduct_ratio': 2,
            'kernel_size': [3, 2],
            'parts': parts,
            'adaptive': False,
            'edge_importance': True
        }

        A_t = torch.from_numpy(A_stack).float().to(device)

        try:
            model = MPGCN(data_shape, args.num_class, A_t, **model_kwargs).to(device)
            model.eval()
        except Exception as e:
            print(f'Error constructing model for {x_path.name}:', e)
            continue

        with torch.no_grad():
            try:
                out, feat = model(data)
            except Exception as e:
                print(f'Forward error for {x_path.name}:', e)
                continue

        logits = out.cpu().numpy().tolist()
        pred = int(out.argmax(dim=1).cpu().item())
        results.append({'file': x_path.name, 'pred': pred, 'logits': logits})
        print(f'[{done+1}] {x_path.name} -> pred={pred} logits_shape={np.array(logits).shape}')
        done += 1

    # Save summary CSV
    import csv
    out_csv = graph_dir / 'forward_results.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'pred', 'logits'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f'Done. Results saved to {out_csv}')


if __name__ == '__main__':
    main()

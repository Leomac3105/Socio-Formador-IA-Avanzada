#!/usr/bin/env python3
"""Evaluate checkpoint and save predictions to CSV.

Usage:
  python Playground/scripts/eval_and_predict.py \
    --graph-dir Playground/data/graph/with_objects \
    --csv Playground/data/graph/with_objects/val.csv \
    --checkpoint Playground/data/graph/with_objects/checkpoint_light_long.pth \
    --out Playground/data/graph/with_objects/predictions_val.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import sys
ROOT = Path(__file__).resolve().parents[2]
MPGCN_PATH = ROOT / '01-replica-paper' / 'MP-GCN'
sys.path.insert(0, str(MPGCN_PATH))

try:
    from src.model.MPGCN.nets import MPGCN
except Exception as e:
    print('Error importing MPGCN:', e)
    raise


def resolve_X_path(graph_dir: Path, fname: str) -> Path:
    p = graph_dir / fname
    if p.exists():
        return p
    stem = Path(fname)
    alt = graph_dir / (stem.stem + '_X.npy')
    if alt.exists():
        return alt
    alt2 = graph_dir / (stem.name + '_X.npy')
    if alt2.exists():
        return alt2
    raise FileNotFoundError(p)


def build_A_from_sample(graph_dir: Path, sample_row_fname: str):
    stem = Path(sample_row_fname)
    cand = graph_dir / (stem.stem + '_A0.npy')
    if not cand.exists():
        cand = graph_dir / (stem.name + '_A0.npy')
    if not cand.exists():
        name_no_x = stem.stem
        if name_no_x.endswith('_X'):
            cand = graph_dir / (name_no_x[:-2] + '_A0.npy')
    if not cand.exists():
        raise FileNotFoundError('A0 adjacency not found for sample: ' + str(stem))
    A0 = np.load(cand)
    mats = [A0]
    base_stem = Path(cand).stem[:-3] if str(cand).endswith('_A0.npy') else Path(cand).stem
    A_intra_p = graph_dir / (base_stem + '_A_intra.npy')
    A_inter_p = graph_dir / (base_stem + '_A_inter.npy')
    for p in [A_intra_p, A_inter_p]:
        if p.exists():
            mats.append(np.load(p))
    A_stack = np.stack(mats, axis=0)
    return torch.tensor(A_stack, dtype=torch.float32)


class GraphDatasetSimple(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, graph_dir: Path, label_encoder: LabelEncoder):
        self.df = df.reset_index(drop=True)
        self.graph_dir = Path(graph_dir)
        self.le = label_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['file']
        Xp = resolve_X_path(self.graph_dir, fname)
        X = np.load(Xp).astype(np.float32)
        label = row.get('label', None)
        y = self.le.transform([label])[0] if label is not None else -1
        return {'X': X, 'y': int(y), 'file': fname}


def collate_fn(batch):
    Xs = [b['X'] for b in batch]
    files = [b['file'] for b in batch]
    ys = torch.tensor([b['y'] for b in batch], dtype=torch.long)
    Xs_t = [torch.from_numpy(x).unsqueeze(0) for x in Xs]
    Xb = torch.stack(Xs_t, dim=0)
    return Xb, ys, files


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph-dir', required=True)
    p.add_argument('--csv', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    graph_dir = Path(args.graph_dir)
    df = pd.read_csv(args.csv)

    # load checkpoint to get label classes
    try:
        ckpt = torch.load(str(args.checkpoint), map_location='cpu')
    except Exception:
        # try loading with weights_only=False (allow pickled objects) if needed
        try:
            ckpt = torch.load(str(args.checkpoint), map_location='cpu', weights_only=False)
        except Exception as e:
            raise
    if isinstance(ckpt, dict) and 'label_classes' in ckpt:
        classes = list(ckpt['label_classes'])
        state = ckpt.get('model_state', None)
    else:
        classes = None
        state = ckpt

    if classes is None:
        le = LabelEncoder()
        le.fit(df['label'].unique())
        classes = list(le.classes_)
    else:
        le = LabelEncoder()
        le.classes_ = np.array(classes)

    # build adjacency from first sample
    sample_fname = df.iloc[0]['file']
    A_stack = build_A_from_sample(graph_dir, sample_fname)

    # infer data shape from one sample
    sample_X_path = resolve_X_path(graph_dir, sample_fname)
    sample_X = np.load(sample_X_path)
    C, T, V, M = sample_X.shape
    data_shape = (1, C, T, V, M)

    model_kwargs = {
        'use_att': True,
        'reduct_ratio': 2,
        'kernel_size': [3, 2],
        'parts': [np.arange(V)],
        'adaptive': False,
        'edge_importance': True,
    }
    num_classes = len(classes)
    model = MPGCN(data_shape=data_shape, num_class=num_classes, A=A_stack, **model_kwargs)
    if state is not None:
        try:
            model.load_state_dict(state)
        except Exception as e:
            print('Warning: could not load state_dict directly:', e)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    ds = GraphDatasetSimple(df, graph_dir, le)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    rows = []
    with torch.no_grad():
        for Xb, ys, files in tqdm(loader, desc='predict'):
            Xb = Xb.to(device)
            out = model(Xb)
            if isinstance(out, (tuple, list)):
                out = out[0]
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1).cpu().tolist()
            logits = out.cpu().numpy().tolist()
            probs_np = probs.cpu().numpy().tolist()
            for i, f in enumerate(files):
                true_idx = int(ys[i].item()) if ys is not None else -1
                true_label = le.inverse_transform([true_idx])[0] if true_idx >= 0 else ''
                pred_idx = int(preds[i])
                pred_label = classes[pred_idx]
                rows.append({'file': f, 'true_label': true_label, 'pred_idx': pred_idx, 'pred_label': pred_label, 'logits': logits[i], 'probs': probs_np[i]})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print('Predictions saved to', args.out)


if __name__ == '__main__':
    main()

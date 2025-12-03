#!/usr/bin/env python3
"""
Light fine-tuning script for MP-GCN using generated graph files.

Features:
- Loads `train.csv` / `val.csv` with columns `file,label[,conf_weight]`.
- Dataset loads `*_X.npy` and uses adjacency `_A0.npy`, `_A_intra.npy`, `_A_inter.npy` from the same stem.
- Builds MPGCN from `01-replica-paper/MP-GCN` codebase and trains for a few epochs.
- Supports `sample_weight` per-sample (column `conf_weight`) and `class_weights` computed from training set.

Usage example:
  python Playground/scripts/train_mpgcn_light.py \
    --graph-dir Playground/data/graph/with_objects \
    --train Playground/data/graph/with_objects/train.csv \
    --val Playground/data/graph/with_objects/val.csv \
    --out checkpoint_light.pth --epochs 3 --batch-size 8

"""
import argparse
import os
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import sys
# ensure MP-GCN repo is importable (relative path to repo)
ROOT = Path(__file__).resolve().parents[2]
MPGCN_PATH = ROOT / '01-replica-paper' / 'MP-GCN'
sys.path.insert(0, str(MPGCN_PATH))

try:
    from src.model.MPGCN.nets import MPGCN
except Exception as e:
    print('Error importing MPGCN from 01-replica-paper/MP-GCN:', e)
    raise


def find_adj_files(stem_path: Path):
    # returns tuple A0, A_intra, A_inter paths if present
    base = stem_path.parent / stem_path.stem
    A0 = base + '_A0.npy'
    A_intra = base + '_A_intra.npy'
    A_inter = base + '_A_inter.npy'
    return A0, A_intra, A_inter


class GraphDataset(Dataset):
    def __init__(self, df: pd.DataFrame, graph_dir: str, label_encoder: LabelEncoder):
        self.df = df.reset_index(drop=True)
        self.graph_dir = Path(graph_dir)
        self.le = label_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['file']
        stem = Path(fname)
        # resolve possible filename variants: prefer exact, then replace .npy -> _X.npy
        Xp = self.graph_dir / fname
        if not Xp.exists():
            alt = self.graph_dir / (stem.stem + '_X.npy')
            if alt.exists():
                Xp = alt
            else:
                alt2 = self.graph_dir / (stem.name + '_X.npy')
                if alt2.exists():
                    Xp = alt2
        if not Xp.exists():
            raise FileNotFoundError(f'X file not found: {Xp} (tried original and _X variants)')
        X = np.load(Xp)
        # expected shape saved: (C, T, V, M)
        # model expects (I, C, T, V, M); we'll add I=1 in collate
        label = row['label']
        y = self.le.transform([label])[0]
        sample_weight = float(row.get('conf_weight', 1.0)) if 'conf_weight' in row.index else 1.0
        # read A from same stem (we'll rely on model A from first sample at init)
        return {'X': X.astype(np.float32), 'y': int(y), 'w': float(sample_weight), 'stem': str(stem)}


def collate_fn(batch):
    # batch: list of dicts
    Xs = [b['X'] for b in batch]
    ys = torch.tensor([b['y'] for b in batch], dtype=torch.long)
    ws = torch.tensor([b['w'] for b in batch], dtype=torch.float32)
    # ensure shapes align: (C,T,V,M) -> add I=1 and batch
    Xs_t = [torch.from_numpy(x).unsqueeze(0) for x in Xs]
    Xb = torch.stack(Xs_t, dim=0)  # (B, I=1, C, T, V, M)
    return Xb, ys, ws


def build_A_from_sample(graph_dir: Path, sample_row_fname: str):
    stem = Path(sample_row_fname)
    # adjacency files are stored as <stem>_A0.npy where <stem> may be either with or without _X
    cand = graph_dir / (stem.stem + '_A0.npy')
    if not cand.exists():
        cand = graph_dir / (stem.name + '_A0.npy')
    if not cand.exists():
        # maybe sample_row_fname already contains _X.npy; try remove '_X' from stem
        name_no_x = stem.stem
        if name_no_x.endswith('_X'):
            cand = graph_dir / (name_no_x[:-2] + '_A0.npy')
    if not cand.exists():
        raise FileNotFoundError('A0 adjacency not found for sample: ' + str(stem))
    A0 = np.load(cand)
    # stack the adjacency (if other parts not present, attempt to infer)
    mats = [A0]
    # try intra/inter variants based on resolved A0 stem
    base_stem = Path(cand).stem[:-3] if str(cand).endswith('_A0.npy') else Path(cand).stem
    A_intra_p = graph_dir / (base_stem + '_A_intra.npy')
    A_inter_p = graph_dir / (base_stem + '_A_inter.npy')
    for p in [A_intra_p, A_inter_p]:
        if p.exists():
            mats.append(np.load(p))
    A_stack = np.stack(mats, axis=0)  # (K, V, V)
    return torch.tensor(A_stack, dtype=torch.float32)


def train_epoch(model, loader, opt, device, class_weights_tensor=None):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for Xb, ys, ws in tqdm(loader, desc='train', leave=False):
        Xb = Xb.to(device)
        ys = ys.to(device)
        ws = ws.to(device)
        opt.zero_grad()
        out = model(Xb)
        if isinstance(out, (tuple, list)):
            out = out[0]
        # compute per-sample loss
        if class_weights_tensor is not None:
            crit = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device), reduction='none')
        else:
            crit = nn.CrossEntropyLoss(reduction='none')
        losses = crit(out, ys)  # (B,)
        weighted = losses * ws
        loss = weighted.mean()
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * Xb.shape[0]
        total_samples += Xb.shape[0]
    return total_loss / max(1, total_samples)


def eval_epoch(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for Xb, ys, ws in tqdm(loader, desc='eval', leave=False):
            Xb = Xb.to(device)
            ys = ys.to(device)
            out = model(Xb)
            if isinstance(out, (tuple, list)):
                out = out[0]
            preds = out.argmax(dim=1)
            correct += (preds == ys).sum().item()
            total += ys.size(0)
            y_true.extend(ys.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    acc = correct / max(1, total)
    return acc, y_true, y_pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph-dir', default='Playground/data/graph/with_objects')
    p.add_argument('--train', default='Playground/data/graph/with_objects/train.csv')
    p.add_argument('--val', default='Playground/data/graph/with_objects/val.csv')
    p.add_argument('--out', default='Playground/data/graph/with_objects/checkpoint_light.pth')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num-workers', type=int, default=4, help='num DataLoader workers')
    p.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    graph_dir = Path(args.graph_dir)
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)

    # label encoder
    le = LabelEncoder()
    le.fit(train_df['label'].unique())

    # dataset + dataloaders
    train_ds = GraphDataset(train_df, graph_dir, le)
    val_ds = GraphDataset(val_df, graph_dir, le)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(1, args.num_workers//2), collate_fn=collate_fn)

    # build A from first training sample to create model (assumes consistent graph sizes)
    sample_fname = train_df.iloc[0]['file']
    A_stack = build_A_from_sample(graph_dir, sample_fname)

    # inspect one sample X to know shapes. Resolve filename variants as in dataset.
    sample_path = graph_dir / sample_fname
    if not sample_path.exists():
        alt = graph_dir / (Path(sample_fname).stem + '_X.npy')
        if alt.exists():
            sample_path = alt
        else:
            alt2 = graph_dir / (Path(sample_fname).name + '_X.npy')
            if alt2.exists():
                sample_path = alt2
    if not sample_path.exists():
        raise FileNotFoundError(f'Could not find sample X for shape inference: tried {sample_path} and _X variants')
    sample_X = np.load(sample_path)
    # sample_X shape: (C, T, V, M)
    C, T, V, M = sample_X.shape
    data_shape = (1, C, T, V, M)

    num_classes = len(le.classes_)
    print(f'Creating model data_shape={data_shape} num_classes={num_classes} device={args.device}')
    # prepare simple parts grouping: single part containing all joints
    parts = [np.arange(V)]
    model_kwargs = {
        'use_att': True,
        'reduct_ratio': 2,
        'kernel_size': [3, 2],
        'parts': parts,
        'adaptive': False,
        'edge_importance': True,
    }
    model = MPGCN(data_shape=data_shape, num_class=num_classes, A=A_stack, **model_kwargs)
    device = torch.device(args.device)
    model = model.to(device)

    # class weights (inverse freq)
    counts = train_df['label'].value_counts().reindex(le.classes_).fillna(0).astype(float)
    inv = 1.0 / (counts + 1e-6)
    class_weights = inv / inv.sum() * len(inv)
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # resume support
    start_epoch = 1
    best_val = 0.0
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print('Resuming from checkpoint:', ckpt_path)
            try:
                ckpt = torch.load(str(ckpt_path), map_location='cpu')
            except Exception as e:
                # try loading with weights_only=False for newer PyTorch security
                try:
                    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
                except Exception as e2:
                    print('Error loading checkpoint:', e2)
                    ckpt = None
            if ckpt is not None:
                try:
                    if 'model_state' in ckpt:
                        model.load_state_dict(ckpt['model_state'])
                    else:
                        # checkpoint might be just a state_dict
                        model.load_state_dict(ckpt)
                    start_epoch = int(ckpt.get('epoch', 0)) + 1 if isinstance(ckpt, dict) else 1
                except Exception as e:
                    print('Warning: could not fully load model_state:', e)
        else:
            print('Warning: resume checkpoint not found:', ckpt_path)

    # prepare metrics logging
    metrics = []
    metrics_out = str(Path(args.out).with_suffix('')) + '.metrics.csv'
    confmat_out = str(Path(args.out).with_suffix('')) + '.confmat.png'

    for epoch in range(start_epoch, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}')
        train_loss = train_epoch(model, train_loader, opt, device, class_weights_tensor=class_weights_tensor)
        val_acc, y_true, y_pred = eval_epoch(model, val_loader, device)
        print(f'  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}')

        # record metrics
        metrics.append({'epoch': epoch, 'train_loss': train_loss, 'val_acc': val_acc})
        # save metrics CSV each epoch
        try:
            import pandas as _pd
            _pd.DataFrame(metrics).to_csv(metrics_out, index=False)
        except Exception:
            # fallback to csv writer
            import csv as _csv
            with open(metrics_out, 'w', newline='') as f:
                writer = _csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_acc'])
                writer.writeheader()
                for m in metrics:
                    writer.writerow(m)

        # confusion matrix + classification report
        try:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
            plt.figure(figsize=(6,6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion matrix')
            plt.colorbar()
            tick_marks = list(range(num_classes))
            plt.xticks(tick_marks, le.classes_, rotation=45)
            plt.yticks(tick_marks, le.classes_)
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             horizontalalignment='center',
                             color='white' if cm[i, j] > thresh else 'black')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            os.makedirs(os.path.dirname(confmat_out), exist_ok=True)
            plt.savefig(confmat_out)
            plt.close()

            # save classification report
            report = classification_report(y_true, y_pred, target_names=list(le.classes_), zero_division=0)
            with open(str(Path(args.out).with_suffix('')) + '.classification_report.txt', 'w') as f:
                f.write(report)
        except Exception as e:
            print('Warning: could not write confusion matrix or report:', e)

        if val_acc > best_val:
            best_val = val_acc
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'label_classes': le.classes_}, args.out)
            print('  Saved best checkpoint to', args.out)

    print('Training complete. Best val acc:', best_val)


if __name__ == '__main__':
    main()

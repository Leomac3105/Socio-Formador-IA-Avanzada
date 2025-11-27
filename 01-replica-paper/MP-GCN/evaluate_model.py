#!/usr/bin/env python3
# Final evaluation script with correct config from trained checkpoint
# Now includes proper shape handling for feeder outputs

import os
import sys
import torch
import json
import numpy as np
from pathlib import Path

# ensure project root is on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print('='*80)
print('MP-GCN Evaluation Script - Using Real Checkpoint Config')
print('='*80)

# Import project modules
from src.model.MPGCN.nets import MPGCN
from src.dataset.graphs import Graph
from src.dataset.volleyball_feeder import Volleyball_Feeder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Config from training
CONFIG = {
    'graph': 'coco_ball-12',
    'labeling': 'intra-inter',
    'inputs': 'JVBM',  # 4 inputs: Joint, Velocity, Bone, bone-Motion
    'input_dims': 2,
    'window': [10, 30],
    'ball': True,
    'person_id': list(range(12)),
    'inter_link': 'pairwise',
    'hop': 1,
    'dilation': 2,
    'use_att': True,
    'reduct_ratio': 2,
    'kernel_size': [3, 2],
    'num_class': 8,  # volleyball has 8 group activity classes
    'batch_size': 16
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Find checkpoint
ckpt_path = 'workdir/volleyball/mpgcn/2025-10-06 16-50-06/MPGCN_volleyball.pth.tar'
if not os.path.exists(ckpt_path):
    ckpt_path = 'workdir/checkpoint.pth.tar'
if not os.path.exists(ckpt_path):
    print(f'ERROR: Checkpoint not found at {ckpt_path}')
    sys.exit(1)
print(f'Checkpoint: {ckpt_path}')

# Build graph
print(f"Building graph: {CONFIG['graph']} with labeling {CONFIG['labeling']}")
graph = Graph(
    dataset='volleyball',
    graph=CONFIG['graph'],
    labeling=CONFIG['labeling'],
    inter_link=CONFIG['inter_link'],
    hop=CONFIG['hop'],
    dilation=CONFIG['dilation']
)
print(f'Graph: num_node={graph.num_node}, num_person={graph.num_person}')

# Build feeder
root_folder = './data/volleyball'
eval_phase_candidates = ['val', 'eval', 'validation']
for candidate in eval_phase_candidates:
    if os.path.exists(os.path.join(root_folder, f'{candidate}_data.npy')):
        eval_phase = candidate
        break
else:
    print(f'ERROR: None of the expected evaluation splits {eval_phase_candidates} found in {root_folder}')
    sys.exit(1)

print(f'Building feeder from: {root_folder} (phase="{eval_phase}")')
try:
    feeder = Volleyball_Feeder(
        phase=eval_phase,
        graph=graph,
        root_folder=root_folder,
        inputs=CONFIG['inputs'],
        debug=False,
        ball=CONFIG['ball'],
        object_folder='./data/volleyball',
        window=CONFIG['window'],
        person_id=CONFIG['person_id'],
        input_dims=CONFIG['input_dims']
    )
    print(f'Feeder created: {len(feeder)} samples')
    data_shape = feeder.get_datashape()
    print(f'Data shape (per sample): {data_shape}')
except Exception as e:
    print(f'ERROR creating feeder: {e}')
    sys.exit(1)

# Inspect one sample to understand shapes
print("="*80)
print("Inspecting one sample for shape verification...")
print("="*80)

data, label, name = feeder[0]
print(f"Sample name: {name}")
print(f"Label: {label}")
print(f"Tensor shape: {data.shape}")

# Handle both 4D and 5D cases safely
if len(data.shape) == 5:
    I, C, T, Vp, M = data.shape
    print(f"Inputs (I): {I}")
    print(f"Channels (C): {C}")
    print(f"Frames (T): {T}")
    print(f"Nodes (V’): {Vp}")
    print(f"Persons (M): {M}")
elif len(data.shape) == 4:
    C, T, Vp, M = data.shape
    print(f"Channels (C): {C}")
    print(f"Frames (T): {T}")
    print(f"Nodes (V’): {Vp}")
    print(f"Persons (M): {M}")
else:
    raise ValueError(f"Unexpected data shape: {data.shape}")

# Create DataLoader
dl = torch.utils.data.DataLoader(feeder, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

# Build model
print('Building MPGCN model...')
A = torch.from_numpy(np.array(graph.A)).float()
model_kwargs = {
    'use_att': CONFIG['use_att'],
    'reduct_ratio': CONFIG['reduct_ratio'],
    'kernel_size': CONFIG['kernel_size'],
    'parts': graph.parts,
    'adaptive': False,
    'edge_importance': True
}
model = MPGCN(data_shape, CONFIG['num_class'], A, **model_kwargs)
print(f'Model created: {sum(p.numel() for p in model.parameters())} parameters')

# Load checkpoint
print('Loading checkpoint...')
try:
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
except Exception:
    pass
state = torch.load(ckpt_path, map_location=device, weights_only=False)
if isinstance(state, dict) and 'state_dict' in state:
    sd = state['state_dict']
elif isinstance(state, dict) and 'model' in state:
    sd = state['model']
else:
    sd = state
new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
model.load_state_dict(new_sd, strict=True)
model.to(device)
model.eval()
print('Checkpoint loaded successfully!')

# Run evaluation
print('\n' + '='*80)
print('Running Evaluation...')
print('='*80)

y_true_all, y_pred_all = [], []

with torch.no_grad():
    for batch_idx, (data, label, name) in enumerate(dl):
        data = data.float().to(device)
        # model expects [B, I, C, T, V, M]
        out, _ = model(data)
        preds = out.argmax(dim=1).cpu().numpy()
        y_true_all.append(label.numpy())
        y_pred_all.append(preds)
        if (batch_idx + 1) % 10 == 0:
            print(f'Processed {(batch_idx + 1) * CONFIG["batch_size"]} / {len(feeder)} samples')

y_true = np.concatenate(y_true_all, axis=0)
y_pred = np.concatenate(y_pred_all, axis=0)

# Compute metrics
top1 = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred, labels=list(range(CONFIG['num_class'])))

print('\n' + '='*80)
print('RESULTS')
print('='*80)
print(f'Top-1 Accuracy: {top1:.4f} ({top1*100:.2f}%)')
print(f'Confusion Matrix Shape: {cm.shape}')
print('\nConfusion Matrix:')
print(cm)

# Save results
output = {
    'top1': float(top1),
    'top1_percent': float(top1 * 100),
    'confusion_matrix': cm.tolist(),
    'y_true': y_true.tolist(),
    'y_pred': y_pred.tolist(),
    'num_samples': int(len(y_true)),
    'num_classes': CONFIG['num_class'],
    'config': CONFIG
}

result_path = 'workdir/evaluation_result.json'
with open(result_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nResults saved to: {result_path}')

# Save confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix - Top-1: {top1:.4f}')
plt.tight_layout()
cm_path = 'workdir/confusion_matrix.png'
plt.savefig(cm_path, dpi=150)
print(f'Confusion matrix plot saved to: {cm_path}')

print('\n' + '='*80)
print('Evaluation Complete!')
print('='*80)

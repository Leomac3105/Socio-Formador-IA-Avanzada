#!/usr/bin/env python3
"""Label windows using a local CLIP zero-shot classifier.

Requiere: `pip install transformers torch pillow` (ver instrucciones abajo).

Entrada: `Playground/data/graph/with_objects/candidate_frames.txt` (creado por `render_window_frames.py`)
Salida: `Playground/data/graph/with_objects/vlm_labels_clip.csv` con columnas `file,label,conf_weight`
"""
from __future__ import annotations

import csv
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel


LABELS = [
    "Transit",
    "Social_People",
    "Play_Object_Normal",
    "Play_Object_Risk",
    "Adult_Assisting",
    "Negative_Contact",
]


def main():
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / 'Playground' / 'data' / 'graph' / 'with_objects' / 'candidate_frames.txt'
    out_csv = repo_root / 'Playground' / 'data' / 'graph' / 'with_objects' / 'vlm_labels_clip.csv'

    if not candidate.exists():
        print('candidate_frames.txt not found:', candidate)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    model_name = 'openai/clip-vit-base-patch32'
    print('Loading CLIP model:', model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    with open(candidate) as f, open(out_csv, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['file','label','conf_weight'])
        for lineno, line in enumerate(f, start=1):
            parts = line.strip().split('\t')
            if len(parts) < 2:
                print(f"Skipping malformed line {lineno} in {candidate}: '{line.strip()}'")
                continue
            npy_path, img_path = parts[0], parts[1]
            img_path_p = Path(img_path)
            if not img_path_p.exists():
                alt = repo_root / img_path
                if alt.exists():
                    img_path_p = alt
            try:
                img = Image.open(img_path_p).convert('RGB')
            except Exception as e:
                print(f"Failed to open image for line {lineno}: {img_path_p} ({e})")
                continue
            inputs = processor(text=LABELS, images=img, return_tensors='pt', padding=True).to(device)
            with torch.no_grad():
                logits_per_image = model(**inputs).logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
            idx = int(probs.argmax())
            label = LABELS[idx]
            conf = float(probs[idx])
            writer.writerow([Path(npy_path).name, label, conf])
            print(Path(npy_path).name, label, f'{conf:.3f}')

    print('Saved labels to', out_csv)


if __name__ == '__main__':
    main()

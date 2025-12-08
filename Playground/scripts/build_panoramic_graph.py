#!/usr/bin/env python3
"""Construye un grafo panorámico básico desde una ventana `.npy` de esqueletos.

Entrada esperada (.npy):
 - Forma: `(T, K_max, 17, 2)` o `(N, T, K_max, 17, 2)` (si hay varias ventanas en el archivo).

Salida:
 - `X.npy`: tensor `X` con shape `(C, T, V', M)` donde C=8 (Jx, Jy, Bx, By, JMx, JMy, BMx, BMy),
   V' = 17 + n_obj, M = K_max.
 - `A0.npy`, `A_intra.npy`, `A_inter.npy`: matrices de adyacencia `V' x V'` (básicas, ver notas).

Notas importantes:
 - Los centroides de objetos deben estar en `configs/objects.yaml` para la `camera_id` proporcionada
   y en coordenadas normalizadas 0..1 (x,y).
 - Indices "manos" y "pelvis" utilizados aquí son aproximados para el esquema de 17 puntos; revisa
   y ajusta `HAND_INDICES` y `PELVIS_INDICES` si usas otro esquema.
 - Este script genera adyacencias simples que sirven como prototipo. El grafo y pesos deben afinarse
   para replicar exactamente la topología descrita en el paper MP-GCN.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import numpy as np


# Ajusta según tu esquema de 17 joints. Estos son índices aproximados y deben revisarse.
# Aquí usamos: wrists ~ [9,10], pelvis ~ [11,12] (como en el script de extracción previo).
HAND_INDICES = [9, 10]
PELVIS_INDICES = [11, 12]


def load_objects(objects_yaml: Path, camera_id: str) -> list[tuple[float, float]]:
    data = yaml.safe_load(objects_yaml.read_text())
    if camera_id not in data:
        raise KeyError(f"Camera id '{camera_id}' not found in {objects_yaml}")
    cam = data[camera_id]
    objs = []
    for o in cam.get("objects", []):
        cx, cy = o.get("centroid", [0.0, 0.0])
        objs.append((float(cx), float(cy)))
    return objs


def ensure_window(arr: np.ndarray) -> np.ndarray:
    # Convertir a forma (T, K, 17, 2) si el npy contiene lote
    if arr.ndim == 5:
        # (N, T, K, 17, 2) -> tomar la primera ventana para prototipo
        return arr[0]
    if arr.ndim == 4:
        return arr
    raise ValueError("Input .npy debe tener shape (T,K,17,2) o (N,T,K,17,2)")


def build_streams(window: np.ndarray, objects: list[tuple[float, float]], pelvis_per_person: np.ndarray | None = None, torso_per_person: np.ndarray | None = None, distance_threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    # window: (T, K, 17, 2)
    T, K, V, _ = window.shape
    n_obj = len(objects)
    Vp = V + n_obj
    M = K

    # Preallocate streams: we'll create J, B, JM, BM each como (2, T, V', M)
    J = np.zeros((2, T, Vp, M), dtype=np.float32)
    B = np.zeros((2, T, Vp, M), dtype=np.float32)

    # Llenar joints (x,y) — ya están normalizados 0..1 respecto a imagen completa
    for m in range(M):
        for v in range(V):
            # window[:, m, v, :] -> (T,2)
            coords = window[:, m, v, :]
            J[0, :, v, m] = coords[:, 0]
            J[1, :, v, m] = coords[:, 1]

    # Objetos: replicar centroides para todos los frames y personas
    eps = 1e-8
    # temporarily store object positions per person per frame (T, M)
    obj_pos = np.zeros((n_obj, 2, T, M), dtype=np.float32)
    for i_obj, (cx, cy) in enumerate(objects):
        v_idx = V + i_obj
        if pelvis_per_person is not None and torso_per_person is not None and pelvis_per_person.shape[0] == M:
            # transform object centroid from image-normalized coords into person-centered, torso-scaled frame
            for m in range(M):
                px, py = float(pelvis_per_person[m, 0]), float(pelvis_per_person[m, 1])
                # Avoid extremely small torso scale which inflates transformed coords.
                # Use a conservative minimum torso scale (in torso-normalized units).
                min_torso = 0.02
                try:
                    s_raw = float(torso_per_person[m])
                except Exception:
                    s_raw = eps
                s = s_raw if s_raw >= min_torso else min_torso
                valx = (cx - px) / s
                valy = (cy - py) / s
                J[0, :, v_idx, m] = valx
                J[1, :, v_idx, m] = valy
                obj_pos[i_obj, 0, :, m] = valx
                obj_pos[i_obj, 1, :, m] = valy
        else:
            # no metadata: replicate raw image-normalized centroid (fallback)
            J[0, :, v_idx, :] = cx
            J[1, :, v_idx, :] = cy
            obj_pos[i_obj, 0, :, :] = cx
            obj_pos[i_obj, 1, :, :] = cy

    # Bones: simple vector respecto al centro del torso (pelvis mean) — alternativa a árbol kinematic
    pelvis = np.zeros((T, M, 2), dtype=np.float32)
    for m in range(M):
        # usar promedio de los puntos PELVIS_INDICES si existen
        pts = window[:, m, :, :]
        # verificar índices dentro de rango
        valid = [i for i in PELVIS_INDICES if i < V]
        if valid:
            pelvis[:, m, :] = pts[:, valid, :].mean(axis=1)
        else:
            pelvis[:, m, :] = pts[:, 0, :]

    for m in range(M):
        for v in range(V):
            coords = window[:, m, v, :]
            vec = coords - pelvis[:, m, :]
            B[0, :, v, m] = vec[:, 0]
            B[1, :, v, m] = vec[:, 1]
        # objetos: vector objeto - pelvis (or transformed values stored in obj_pos)
        for i_obj, (cx, cy) in enumerate(objects):
            v_idx = V + i_obj
            # obj_pos shape (n_obj,2,T,M)
            B[0, :, v_idx, m] = obj_pos[i_obj, 0, :, m] - 0.0
            B[1, :, v_idx, m] = obj_pos[i_obj, 1, :, m] - 0.0

    # If requested, compute proximity mask and zero-out object channels for non-interacting persons
    if distance_threshold is not None:
        # compute per-person-object min distance to nearest hand across frames
        interaction = np.zeros((n_obj, M), dtype=bool)
        for i_obj in range(n_obj):
            for m in range(M):
                # distance to each hand index over frames
                min_dist = float('inf')
                for h in HAND_INDICES:
                    if h >= V:
                        continue
                    # window[:,m,h,:] is (T,2) in person-centered units
                    hand = window[:, m, h, :]
                    obj = obj_pos[i_obj, :, :, m].T  # shape (T,2)
                    d = np.linalg.norm(hand - obj, axis=1)
                    min_dist = min(min_dist, float(np.min(d)))
                if min_dist <= distance_threshold:
                    interaction[i_obj, m] = True
        # zero-out object entries for (i_obj,m) where interaction False
        for i_obj in range(n_obj):
            v_idx = V + i_obj
            for m in range(M):
                if not interaction[i_obj, m]:
                    J[:, :, v_idx, m] = 0.0
                    B[:, :, v_idx, m] = 0.0

    # JM / BM: diferencias temporales (t -> t+1), último frame 0
    JM = np.zeros_like(J)
    BM = np.zeros_like(B)
    JM[:, :-1, :, :] = J[:, 1:, :, :] - J[:, :-1, :, :]
    BM[:, :-1, :, :] = B[:, 1:, :, :] - B[:, :-1, :, :]

    # Concatenate streams into X: [Jx,Jy,Bx,By,JMx,JMy,BMx,BMy] -> C=8
    X = np.concatenate([J[0:1], J[1:2], B[0:1], B[1:2], JM[0:1], JM[1:2], BM[0:1], BM[1:2]], axis=0)
    # After concat, each slice has shape (1, T, Vp, M) so X shape = (8, T, Vp, M)
    return X.astype(np.float32), Vp


def build_adjacencies(Vp: int, n_obj: int, V: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # A0: identidad
    A0 = np.eye(Vp, dtype=np.float32)
    A_intra = np.zeros((Vp, Vp), dtype=np.float32)
    A_inter = np.zeros((Vp, Vp), dtype=np.float32)

    # Indices: joints 0..V-1, objects V..Vp-1
    # Conectar manos (si existen) con objetos (intra)
    for h in HAND_INDICES:
        if h >= V:
            continue
        for o in range(n_obj):
            oi = V + o
            A_intra[h, oi] = 1.0
            A_intra[oi, h] = 1.0

    # Conectar pelvis juntas (inter) — promedio de PELVIS_INDICES
    # Aquí marcamos pelvis node (usamos primer pelvis index válido)
    pelvis_idx = None
    for p in PELVIS_INDICES:
        if p < V:
            pelvis_idx = p
            break
    if pelvis_idx is not None:
        A_inter[pelvis_idx, pelvis_idx] = 1.0

    return A0, A_intra, A_inter


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir grafo panorámico desde ventana .npy")
    parser.add_argument("--npy", type=Path, required=True, help="Ruta al .npy de ventana")
    parser.add_argument("--camera-id", type=str, required=True, help="ID de cámara (clave en objects.yaml)")
    parser.add_argument("--objects", type=Path, default=Path("configs/objects.yaml"), help="Ruta a objects.yaml")
    parser.add_argument("--out-dir", type=Path, default=Path("data/graph"), help="Directorio de salida")
    parser.add_argument("--distance-threshold", type=float, default=0.25, help="Umbral de distancia (en unidades torso-scale) para considerar que una persona usa un objeto")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    arr = np.load(args.npy)
    window = ensure_window(arr)
    T, K, V, _ = window.shape

    objects = load_objects(args.objects, args.camera_id)
    # Attempt to locate metadata file alongside the .npy
    stem = Path(args.npy).stem
    meta_path = Path(args.npy).with_suffix("")
    meta_path = meta_path.parent / f"{stem}_meta.npz"
    pelvis_meta = None
    torso_meta = None
    if meta_path.exists():
        try:
            m = np.load(meta_path)
            pelvis_meta = m["pelvis"] if "pelvis" in m else None
            torso_meta = m["torso"] if "torso" in m else None
            print(f"Cargando metadata desde {meta_path}")
        except Exception as e:
            print(f"No se pudo leer metadata {meta_path}: {e}")

    X, Vp = build_streams(window, objects, pelvis_per_person=pelvis_meta, torso_per_person=torso_meta, distance_threshold=args.distance_threshold)
    A0, A_intra, A_inter = build_adjacencies(Vp, len(objects), V)

    # Guardar
    stem = Path(args.npy).stem
    out_X = args.out_dir / f"{stem}_X.npy"
    out_A0 = args.out_dir / f"{stem}_A0.npy"
    out_A_intra = args.out_dir / f"{stem}_A_intra.npy"
    out_A_inter = args.out_dir / f"{stem}_A_inter.npy"

    np.save(out_X, X)
    np.save(out_A0, A0)
    np.save(out_A_intra, A_intra)
    np.save(out_A_inter, A_inter)

    print(f"Guardado: {out_X}")
    print(f"X shape: {X.shape}  (C, T, V', M)")
    print(f"A0 shape: {A0.shape}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extrae esqueletos normalizados y ventanas [T,K_max,17,2] desde videos filtrados."""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "OpenCV (cv2) no está instalado. Ejecuta `pip install opencv-python` "
        "en el mismo entorno antes de correr este script."
    ) from exc

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "MediaPipe no está instalado. Ejecuta `pip install mediapipe` en el entorno activo."
    ) from exc

import numpy as np
import pandas as pd

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "No se encontró Ultralytics YOLO. Instálalo con `pip install ultralytics`."
    ) from exc


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)


def load_video_12fps(video_path: Path, target_fps: int = 12) -> List[np.ndarray]:
    """Carga frames muestreados a ~12 FPS para acelerar la inferencia."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30
    frame_step = max(int(round(fps / target_fps)), 1)

    frames: List[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames


def run_yolo(frame: np.ndarray, model: YOLO) -> List[Tuple[float, float, float, float]]:
    """Detecta personas en un frame usando YOLOv8."""
    detections: List[Tuple[float, float, float, float]] = []
    results = model(frame, verbose=False)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls != 0:  # clase 0 = persona
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append((x1, y1, x2, y2))
    return detections


@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]


class SimpleTracker:
    """Tracker mínimo para mantener IDs consistentes en ventanas cortas."""

    def __init__(self) -> None:
        self.next_id = 0

    def update(self, detections: Sequence[Tuple[float, float, float, float]]) -> List[Track]:
        tracks: List[Track] = []
        for det in detections:
            tracks.append(Track(self.next_id, det))
            self.next_id += 1
        return tracks


def run_pose(frame: np.ndarray, bbox: Tuple[float, float, float, float], pose_estimator) -> np.ndarray:
    """Ejecuta MediaPipe Pose sobre un recorte y devuelve 17 keypoints 2D normalizados (0..1)
    respecto a la imagen completa.

    MediaPipe devuelve landmarks normalizados respecto al `crop` (0..1). Aquí remapeamos cada
    landmark al sistema de coordenadas de la imagen completa usando el `bbox` (x1,y1,x2,y2).
    Esto facilita el emparejamiento con centroides de objetos normalizados en `objects.yaml`.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_img, w_img = frame.shape[:2]

    # Clamp bbox dentro de la imagen
    x1 = max(0, min(x1, w_img - 1))
    x2 = max(0, min(x2, w_img))
    y1 = max(0, min(y1, h_img - 1))
    y2 = max(0, min(y2, h_img))

    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((17, 2), dtype=np.float32)

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    res = pose_estimator.process(crop_rgb)
    keypoints = np.zeros((17, 2), dtype=np.float32)

    if res.pose_landmarks:
        for idx, lm in enumerate(res.pose_landmarks.landmark[:17]):
            # lm.x/lm.y are normalized w.r.t. crop. Convert to image pixel coords then normalize by image size.
            x_img = lm.x * crop_w + x1
            y_img = lm.y * crop_h + y1
            # normalize to 0..1 over full image
            keypoints[idx, 0] = float(x_img / w_img)
            keypoints[idx, 1] = float(y_img / h_img)

            # clamp to [0,1]
            keypoints[idx, 0] = min(max(keypoints[idx, 0], 0.0), 1.0)
            keypoints[idx, 1] = min(max(keypoints[idx, 1], 0.0), 1.0)

    return keypoints


def normalize_skeleton(keypoints: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Traslada al origen (pelvis) y escala usando el torso."""
    kps = keypoints.astype(np.float32).copy()
    pelvis = (kps[11] + kps[12]) / 2
    kps -= pelvis

    shoulders_center = (kps[5] + kps[6]) / 2
    torso_len = np.linalg.norm(shoulders_center) + eps
    kps /= torso_len
    return kps


def extract_skeletons(
    frames: Sequence[np.ndarray],
    model: YOLO,
    pose_estimator,
) -> List[dict]:
    """Genera lista de esqueletos detectados por frame."""
    tracker = SimpleTracker()
    records: List[dict] = []

    for frame_idx, frame in enumerate(frames):
        detections = run_yolo(frame, model)
        tracks = tracker.update(detections)

        for track in tracks:
            keypoints = run_pose(frame, track.bbox, pose_estimator)
            kps_norm = normalize_skeleton(keypoints)
            records.append(
                {
                    "local_frame": frame_idx,
                    "track_id": track.track_id,
                    "keypoints": kps_norm,
                }
            )
    return records


def build_windows(
    skeletons: Sequence[dict],
    window: int = 48,
    k_max: int = 4,
) -> np.ndarray:
    """Agrupa esqueletos normalizados en tensores [N,T,K_max,17,2]."""
    if not skeletons:
        return np.empty((0, window, k_max, 17, 2), dtype=np.float32)

    by_track: dict[int, List[dict]] = {}
    for record in skeletons:
        by_track.setdefault(record["track_id"], []).append(record)
    for records in by_track.values():
        records.sort(key=lambda x: x["local_frame"])

    frame_indices = sorted(record["local_frame"] for record in skeletons)
    start_frame = frame_indices[0]
    end_frame = frame_indices[-1]

    stride = window // 2
    windows: List[np.ndarray] = []

    frame_range = range(start_frame, end_frame + 1)
    for start in frame_range:
        stop = start + window
        if stop - 1 > end_frame:
            break

        tensor = np.zeros((window, k_max, 17, 2), dtype=np.float32)
        track_scores: List[Tuple[int, int]] = []
        for tid, records in by_track.items():
            count = sum(start <= rec["local_frame"] < stop for rec in records)
            if count:
                track_scores.append((count, tid))

        track_scores.sort(reverse=True)
        selected = [tid for _, tid in track_scores[:k_max]]

        for slot, tid in enumerate(selected):
            frame_to_kps = {
                rec["local_frame"]: rec["keypoints"] for rec in by_track[tid]
            }
            for step, frame_id in enumerate(range(start, stop)):
                if frame_id in frame_to_kps:
                    tensor[step, slot] = frame_to_kps[frame_id]

        windows.append(tensor)
        start += stride - 1  # compensate for loop increment

    return np.stack(windows, axis=0) if windows else np.empty((0, window, k_max, 17, 2), dtype=np.float32)


def process_scene(
    row: pd.Series,
    downloads_dir: Path,
    output_dir: Path,
    model: YOLO,
    pose_estimator,
    window: int,
    k_max: int,
) -> int:
    """Procesa una fila de videos.csv y guarda npy por ventana."""
    video_id = row["video_id"]
    blob_name = Path(row["blob_path"]).name
    video_path = downloads_dir / blob_name

    if not video_path.exists():
        logging.warning("Video no encontrado: %s", video_path)
        return 0

    frames = load_video_12fps(video_path)
    if not frames:
        logging.warning("Video sin frames válidos: %s", video_path)
        return 0

    skeletons = extract_skeletons(frames, model, pose_estimator)
    windows = build_windows(skeletons, window=window, k_max=k_max)

    saved = 0
    for idx, tensor in enumerate(windows):
        out_path = output_dir / f"{video_id}_win{idx:03d}.npy"
        np.save(out_path, tensor.astype(np.float32))
        saved += 1

    logging.info("Procesado %s -> %d ventanas", video_id, saved)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrae tensores [T,K_max,17,2] desde videos filtrados.")
    parser.add_argument("--videos-csv", type=Path, default=Path("data/videos.csv"))
    parser.add_argument("--downloads-dir", type=Path, default=Path("Downloads"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/npy"))
    parser.add_argument("--yolo-weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--window", type=int, default=48, help="Número de frames por ventana")
    parser.add_argument("--k-max", type=int, default=4, help="Máximo de personas por ventana")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos_csv = args.videos_csv
    downloads_dir = args.downloads_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not videos_csv.exists():
        raise FileNotFoundError(f"No se encontró {videos_csv}")

    logging.info("Leyendo %s", videos_csv)
    df_videos = pd.read_csv(videos_csv)

    logging.info("Cargando YOLOv8 desde %s", args.yolo_weights)
    model = YOLO(str(args.yolo_weights))
    pose = mp.solutions.pose.Pose(static_image_mode=True)

    total_windows = 0
    for _, row in df_videos.iterrows():
        total_windows += process_scene(
            row=row,
            downloads_dir=downloads_dir,
            output_dir=output_dir,
            model=model,
            pose_estimator=pose,
            window=args.window,
            k_max=args.k_max,
        )

    logging.info("Listo. Ventanas generadas: %d", total_windows)


if __name__ == "__main__":
    main()

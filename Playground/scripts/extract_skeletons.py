#!/usr/bin/env python3
"""
Extrae esqueletos normalizados y ventanas limpias [T,K_max,17,2].

"""

from __future__ import annotations
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from ultralytics import YOLO


# ---------------------------------------------------------
# CONFIG LOGGING
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)


# ---------------------------------------------------------
#   VIDEO LOADER (12 FPS)
# ---------------------------------------------------------
def load_video_12fps(video_path: Path, target_fps: int = 12) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30
    frame_step = max(int(round(fps / target_fps)), 1)

    frames = []
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


# ---------------------------------------------------------
# YOLO DETECTION
# ---------------------------------------------------------
def run_yolo(frame: np.ndarray, model: YOLO) -> List[Tuple[float, float, float, float]]:
    detections = []
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls != 0:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append((x1, y1, x2, y2))

    return detections


# ---------------------------------------------------------
# SIMPLE TRACKER (estable para ventanas cortas)
# ---------------------------------------------------------
@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]


class SimpleTracker:
    def __init__(self):
        self.next_id = 0

    def update(self, detections):
        tracks = []
        for det in detections:
            tracks.append(Track(self.next_id, det))
            self.next_id += 1
        return tracks


# ---------------------------------------------------------
# POSE ESTIMATION (MediaPipe) con remapeo
# ---------------------------------------------------------
def run_pose(frame, bbox, pose_estimator):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((17, 2), dtype=np.float32)

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    res = pose_estimator.process(crop_rgb)

    keypoints = np.zeros((17, 2), dtype=np.float32)
    crop_h, crop_w = crop.shape[:2]

    if res.pose_landmarks:
        for i, lm in enumerate(res.pose_landmarks.landmark[:17]):
            xp = lm.x * crop_w + x1
            yp = lm.y * crop_h + y1
            keypoints[i, 0] = np.clip(xp / w, 0, 1)
            keypoints[i, 1] = np.clip(yp / h, 0, 1)

    return keypoints


# ---------------------------------------------------------
# NORMALIZACIÓN PELVIS → ORIGEN y ESCALA TORSO
# ---------------------------------------------------------
def normalize_skeleton(kps, eps=1e-6):
    k = kps.copy()
    pelvis = (k[11] + k[12]) / 2
    k -= pelvis
    shoulders = (k[5] + k[6]) / 2
    torso = np.linalg.norm(shoulders) + eps
    k /= torso
    return k.astype(np.float32)


# ---------------------------------------------------------
#   EXTRACT SKELETONS (core)
# ---------------------------------------------------------
def extract_skeletons(frames, model, pose_estimator):
    tracker = SimpleTracker()
    records = []

    for fidx, frame in enumerate(frames):
        detections = run_yolo(frame, model)
        tracks = tracker.update(detections)

        for tr in tracks:
            raw = run_pose(frame, tr.bbox, pose_estimator)
            pelvis_img = (raw[11] + raw[12]) / 2
            shoulders = (raw[5] + raw[6]) / 2
            torso_len = float(np.linalg.norm(shoulders - pelvis_img))
            norm_kps = normalize_skeleton(raw)

            records.append(
                {
                    "local_frame": fidx,
                    "track_id": tr.track_id,
                    "keypoints": norm_kps,
                    "raw_keypoints": raw,
                    "pelvis_image": pelvis_img,
                    "torso_scale": max(torso_len, 1e-6),
                }
            )

    return records


# ---------------------------------------------------------
# WINDOW BUILDER (corregido)
# ---------------------------------------------------------
def build_windows(skeletons, window=48, k_max=4, min_frame_frac=0.25):

    if len(skeletons) == 0:
        return np.empty((0, window, k_max, 17, 2), dtype=np.float32), []

    by_track = {}
    for rec in skeletons:
        by_track.setdefault(rec["track_id"], []).append(rec)

    for v in by_track.values():
        v.sort(key=lambda x: x["local_frame"])

    frames = sorted(rec["local_frame"] for rec in skeletons)
    start_f = frames[0]
    end_f = frames[-1]

    stride = window // 2
    min_frames = max(1, int(window * min_frame_frac))

    all_tensors = []
    metas_all = []

    for start in range(start_f, end_f + 1):
        stop = start + window
        if stop > end_f + 1:
            break

        tensor = np.zeros((window, k_max, 17, 2), dtype=np.float32)
        pelvis_slots = np.zeros((k_max, 2), dtype=np.float32)
        torso_slots = np.ones((k_max,), dtype=np.float32)
        slots_ids = [-1] * k_max

        # seleccionar tracks más presentes
        counts = []
        for tid, recs in by_track.items():
            c = sum(start <= r["local_frame"] < stop for r in recs)
            if c > 0:
                counts.append((c, tid))

        counts.sort(reverse=True)
        selected = [tid for _, tid in counts[:k_max]]

        for slot, tid in enumerate(selected):
            recs = by_track[tid]
            frames_map = {r["local_frame"]: r for r in recs}

            present = []
            for tstep, fr in enumerate(range(start, stop)):
                if fr in frames_map:
                    tensor[tstep, slot] = frames_map[fr]["keypoints"]
                    present.append(tstep)

            # stats
            used = [r for r in recs if start <= r["local_frame"] < stop]
            if used:
                pelvis_slots[slot] = np.mean([u["pelvis_image"] for u in used], axis=0)
                torso_slots[slot] = float(
                    max(np.mean([u["torso_scale"] for u in used]), 1e-6)
                )
            slots_ids[slot] = tid

            # rellenado si faltan frames
            if len(present) == 0:
                continue

            if len(present) < min_frames:
                present = np.array(present)
                full = np.arange(window)
                vals = tensor[:, slot]
                available = vals[present]
                dists = np.abs(full[:, None] - present[None, :])
                nearest = dists.argmin(axis=1)
                tensor[:, slot] = available[nearest]

        all_tensors.append(tensor)
        metas_all.append(
            {
                "pelvis": pelvis_slots.astype(np.float32),
                "torso": torso_slots.astype(np.float32),
                "track_ids": np.array(slots_ids, dtype=np.int32),
            }
        )

    if len(all_tensors) == 0:
        return np.empty((0, window, k_max, 17, 2), dtype=np.float32), metas_all

    return np.stack(all_tensors, axis=0), metas_all


# ---------------------------------------------------------
# PROCESS SCENE
# ---------------------------------------------------------
def process_scene(row, downloads, outdir, model, pose, window, k_max, min_frame_frac):
    video_id = row["video_id"]
    blob = Path(row["blob_path"]).name
    vid_path = downloads / blob

    if not vid_path.exists():
        logging.warning("Video no encontrado %s", vid_path)
        return 0

    frames = load_video_12fps(vid_path)
    if len(frames) == 0:
        return 0

    sk = extract_skeletons(frames, model, pose)
    win, meta = build_windows(sk, window=window, k_max=k_max, min_frame_frac=min_frame_frac)

    saved = 0
    for i, tensor in enumerate(win):
        np.save(outdir / f"{video_id}_win{i:03d}.npy", tensor.astype(np.float32))
        np.savez_compressed(
            outdir / f"{video_id}_win{i:03d}_meta.npz",
            pelvis=meta[i]["pelvis"],
            torso=meta[i]["torso"],
            track_ids=meta[i]["track_ids"],
        )
        saved += 1

    return saved


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos-csv", type=Path, default=Path("data/videos.csv"))
    p.add_argument("--downloads-dir", type=Path, default=Path("Downloads"))
    p.add_argument("--output-dir", type=Path, default=Path("data/npy_filled"))
    p.add_argument("--yolo-weights", type=Path, default=Path("yolov8n.pt"))
    p.add_argument("--window", type=int, default=48)
    p.add_argument("--k-max", type=int, default=4)
    p.add_argument("--min-frame-frac", type=float, default=0.25)
    return p.parse_args()


def main():
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.videos_csv)
    model = YOLO(str(args.yolo_weights))
    pose = mp.solutions.pose.Pose(static_image_mode=True)

    total = 0
    for _, row in df.iterrows():
        total += process_scene(
            row=row,
            downloads=args.downloads_dir,
            outdir=out,
            model=model,
            pose=pose,
            window=args.window,
            k_max=args.k_max,
            min_frame_frac=args.min_frame_frac,
        )

    logging.info("LISTO. Ventanas generadas: %d", total)


if __name__ == "__main__":
    main()

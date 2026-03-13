#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

import lidar_utils

# Airbus train RGB mapping from the toolkit README.
RGB_TO_CLASS = {
    (38, 23, 180): (0, "Antenna"),
    (177, 132, 47): (1, "Cable"),
    (129, 81, 97): (2, "Electric Pole"),
    (66, 132, 9): (3, "Wind Turbine"),
}

CLASS_ORDER = ["Antenna", "Cable", "Electric Pole", "Wind Turbine"]
CLASS_NAME_TO_ID = {v: k for _, (k, v) in RGB_TO_CLASS.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Airbus LiDAR heuristic detector v6")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--point-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize-pose-index", type=int, default=None)
    parser.add_argument("--diagnostics-json", type=Path, default=None)
    return parser.parse_args()


def load_h5_as_dataframe(h5_path: Path) -> pd.DataFrame:
    if hasattr(lidar_utils, "load_h5_data"):
        df = lidar_utils.load_h5_data(str(h5_path))
        if isinstance(df, pd.DataFrame):
            return df
    with h5py.File(h5_path, "r") as f:
        if "lidar_points" not in f:
            raise KeyError(f"'lidar_points' not found in {h5_path}")
        data = f["lidar_points"][:]
    return pd.DataFrame(data)


def get_unique_poses_df(df: pd.DataFrame) -> pd.DataFrame:
    pose_cols = ["ego_x", "ego_y", "ego_z", "ego_yaw"]
    if hasattr(lidar_utils, "get_unique_poses"):
        poses = lidar_utils.get_unique_poses(df)
        if isinstance(poses, pd.DataFrame):
            return poses.reset_index(drop=True)
        if isinstance(poses, np.ndarray):
            return pd.DataFrame(poses, columns=pose_cols)
    return df[pose_cols].drop_duplicates().reset_index(drop=True)


def filter_by_pose_df(df: pd.DataFrame, pose_row: pd.Series) -> pd.DataFrame:
    if hasattr(lidar_utils, "filter_by_pose"):
        try:
            out = lidar_utils.filter_by_pose(df, pose_row)
            if isinstance(out, pd.DataFrame):
                return out.reset_index(drop=True)
        except Exception:
            pass
    mask = (
        (df["ego_x"] == pose_row["ego_x"]) &
        (df["ego_y"] == pose_row["ego_y"]) &
        (df["ego_z"] == pose_row["ego_z"]) &
        (df["ego_yaw"] == pose_row["ego_yaw"])
    )
    return df.loc[mask].reset_index(drop=True)


def spherical_to_xyz_robust(frame_df: pd.DataFrame) -> np.ndarray:
    fn = lidar_utils.spherical_to_local_cartesian
    try:
        n_params = len(inspect.signature(fn).parameters)
    except Exception:
        n_params = None
    if n_params == 1:
        xyz = fn(frame_df)
    elif n_params == 3:
        xyz = fn(
            frame_df["distance_cm"].to_numpy(),
            frame_df["azimuth_raw"].to_numpy(),
            frame_df["elevation_raw"].to_numpy(),
        )
    else:
        try:
            xyz = fn(frame_df)
        except TypeError:
            xyz = fn(
                frame_df["distance_cm"].to_numpy(),
                frame_df["azimuth_raw"].to_numpy(),
                frame_df["elevation_raw"].to_numpy(),
            )
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Invalid xyz shape: {xyz.shape}")
    return xyz


def sample_fraction(df: pd.DataFrame, point_fraction: float, seed: int) -> pd.DataFrame:
    if point_fraction >= 0.999:
        return df.reset_index(drop=True)
    n = len(df)
    keep = max(1, int(n * point_fraction))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=keep, replace=False)
    idx.sort()
    return df.iloc[idx].reset_index(drop=True)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0,), dtype=np.int64)
    coords = np.floor(points / voxel_size).astype(np.int32)
    _, first_idx = np.unique(coords, axis=0, return_index=True)
    return np.sort(first_idx)


def class_mask_from_rgb(frame_df: pd.DataFrame, class_name: str) -> np.ndarray:
    target = None
    for rgb, (_, name) in RGB_TO_CLASS.items():
        if name == class_name:
            target = np.array(rgb, dtype=np.int32)
            break
    if target is None:
        raise KeyError(class_name)
    rgb = np.stack(
        [
            frame_df["r"].to_numpy().astype(np.int32),
            frame_df["g"].to_numpy().astype(np.int32),
            frame_df["b"].to_numpy().astype(np.int32),
        ],
        axis=1,
    )
    return np.all(rgb == target[None, :], axis=1)


def pca_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    return center, vals, vecs


def oriented_box_from_points(points: np.ndarray, clamp_cable: bool = False) -> Optional[Dict[str, float]]:
    if len(points) < 5:
        return None
    center, vals, vecs = pca_axes(points)
    axis0 = vecs[:, 0]
    yaw = float(np.arctan2(axis0[1], axis0[0]))
    rot = np.array(
        [[np.cos(-yaw), -np.sin(-yaw), 0.0], [np.sin(-yaw), np.cos(-yaw), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    local = (points - center) @ rot.T
    mins = local.min(axis=0)
    maxs = local.max(axis=0)
    extents = np.maximum(maxs - mins, 0.05)
    width = float(extents[0])
    length = float(extents[1])
    height = float(extents[2])
    if clamp_cable:
        # Keep cable boxes thin but not unrealistically flat.
        width = float(np.clip(min(width, length), 0.05, 0.18))
        height = float(np.clip(height, 0.08, 1.5))
        length = float(max(length, width))
    return {
        "bbox_center_x": float(center[0]),
        "bbox_center_y": float(center[1]),
        "bbox_center_z": float(center[2]),
        "bbox_width": width,
        "bbox_length": length,
        "bbox_height": height,
        "bbox_yaw": yaw,
    }


def merge_boxes(boxes: List[Dict[str, float]], class_name: str) -> List[Dict[str, float]]:
    if len(boxes) <= 1:
        return boxes
    used = np.zeros(len(boxes), dtype=bool)
    merged: List[Dict[str, float]] = []
    for i, bi in enumerate(boxes):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        ci = np.array([bi["bbox_center_x"], bi["bbox_center_y"], bi["bbox_center_z"]])
        yi = bi["bbox_yaw"]
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            bj = boxes[j]
            cj = np.array([bj["bbox_center_x"], bj["bbox_center_y"], bj["bbox_center_z"]])
            yaw_diff = abs(math.atan2(math.sin(yi - bj["bbox_yaw"]), math.cos(yi - bj["bbox_yaw"])))
            dist = np.linalg.norm(ci - cj)
            if class_name == "Cable":
                ok = yaw_diff < 0.2 and dist < 10.0
            elif class_name == "Wind Turbine":
                ok = dist < 20.0
            else:
                ok = yaw_diff < 0.35 and dist < 4.0
            if ok:
                group.append(j)
                used[j] = True
        if len(group) == 1:
            merged.append(bi)
            continue
        pts = []
        for k in group:
            b = boxes[k]
            pts.append([b["bbox_center_x"], b["bbox_center_y"], b["bbox_center_z"]])
        pts_np = np.asarray(pts, dtype=np.float32)
        out = oriented_box_from_points(pts_np, clamp_cable=(class_name == "Cable"))
        if out is None:
            merged.append(bi)
        else:
            if class_name == "Wind Turbine":
                out["bbox_height"] = max(max(boxes[k]["bbox_height"] for k in group), out["bbox_height"])
                out["bbox_width"] = max(max(boxes[k]["bbox_width"] for k in group), out["bbox_width"])
                out["bbox_length"] = max(max(boxes[k]["bbox_length"] for k in group), out["bbox_length"])
            if class_name == "Cable":
                out["bbox_length"] = max(sum(boxes[k]["bbox_length"] for k in group) / max(1, len(group)), out["bbox_length"])
                out["bbox_width"] = min(min(boxes[k]["bbox_width"] for k in group), 0.18)
                out["bbox_height"] = max(boxes[k]["bbox_height"] for k in group)
            merged.append(out)
    return merged


def split_cable_by_axis(points: np.ndarray, gap_threshold: float = 2.5) -> List[np.ndarray]:
    if len(points) < 8:
        return [points]
    center, vals, vecs = pca_axes(points)
    axis = vecs[:, 0]
    proj = (points - center) @ axis
    order = np.argsort(proj)
    proj_sorted = proj[order]
    gaps = np.diff(proj_sorted)
    cut_idx = np.where(gaps > gap_threshold)[0]
    if len(cut_idx) == 0:
        return [points]
    start = 0
    segments = []
    for c in cut_idx:
        seg = points[order[start:c + 1]]
        if len(seg) >= 5:
            segments.append(seg)
        start = c + 1
    last = points[order[start:]]
    if len(last) >= 5:
        segments.append(last)
    return segments if segments else [points]


def detect_cable_boxes(points: np.ndarray, point_fraction: float, diag: Dict) -> List[Dict[str, float]]:
    if len(points) < 10:
        return []
    keep = voxel_downsample(points, voxel_size=0.10 if point_fraction >= 0.5 else 0.08)
    pts = points[keep]
    eps = 1.4 if point_fraction >= 0.75 else 1.6 if point_fraction >= 0.5 else 1.8
    min_samples = 8 if point_fraction >= 0.5 else 6
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    diag.update({"dbscan_eps": eps, "dbscan_min_samples": min_samples, "post_voxel_points": int(len(pts))})
    boxes: List[Dict[str, float]] = []
    raw_clusters = 0
    accepted = 0
    for cid in np.unique(labels):
        if cid < 0:
            continue
        raw_clusters += 1
        cluster = pts[labels == cid]
        if len(cluster) < min_samples:
            continue
        center, vals, vecs = pca_axes(cluster)
        anisotropy = float(vals[0] / max(vals[1], 1e-6))
        vertical_spread = float(cluster[:, 2].max() - cluster[:, 2].min())
        if anisotropy < 8.0:
            continue
        for seg in split_cable_by_axis(cluster, gap_threshold=2.5 if point_fraction >= 0.5 else 3.5):
            if len(seg) < min_samples:
                continue
            box = oriented_box_from_points(seg, clamp_cable=True)
            if box is None:
                continue
            if box["bbox_length"] < 4.0:
                continue
            if box["bbox_width"] > 0.25:
                continue
            if vertical_spread > 3.0:
                # cables should stay thin vertically too
                box["bbox_height"] = min(box["bbox_height"], 1.2)
            accepted += 1
            boxes.append(box)
    merged = merge_boxes(boxes, "Cable")
    diag.update({
        "dbscan_clusters": raw_clusters,
        "accepted_clusters": accepted,
        "merged_clusters": int(len(merged)),
    })
    return merged


def detect_vertical_boxes(points: np.ndarray, class_name: str, point_fraction: float, diag: Dict) -> List[Dict[str, float]]:
    if len(points) < 8:
        return []
    voxel = 0.18 if class_name == "Antenna" else 0.22
    pts = points[voxel_downsample(points, voxel)]
    eps = 0.9 if point_fraction >= 0.5 else 1.1
    min_samples = 8 if class_name == "Antenna" else 10
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    diag.update({"dbscan_eps": eps, "dbscan_min_samples": min_samples, "post_voxel_points": int(len(pts))})
    boxes = []
    raw_clusters = 0
    accepted = 0
    for cid in np.unique(labels):
        if cid < 0:
            continue
        raw_clusters += 1
        cluster = pts[labels == cid]
        if len(cluster) < min_samples:
            continue
        box = oriented_box_from_points(cluster)
        if box is None:
            continue
        horiz = max(box["bbox_width"], box["bbox_length"])
        vertical_ratio = box["bbox_height"] / max(horiz, 1e-3)
        if vertical_ratio < (2.5 if class_name == "Electric Pole" else 1.5):
            continue
        if class_name == "Electric Pole":
            if box["bbox_height"] < 4.0 or horiz > 2.5:
                continue
        else:
            if box["bbox_height"] < 1.0 or horiz > 3.0:
                continue
        accepted += 1
        boxes.append(box)
    merged = merge_boxes(boxes, class_name)
    diag.update({
        "dbscan_clusters": raw_clusters,
        "accepted_clusters": accepted,
        "merged_clusters": int(len(merged)),
    })
    return merged


def detect_turbine_boxes(points: np.ndarray, point_fraction: float, diag: Dict) -> List[Dict[str, float]]:
    if len(points) < 20:
        return []
    pts = points[voxel_downsample(points, 0.40)]
    eps = 2.0 if point_fraction >= 0.5 else 2.3
    min_samples = 12
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    diag.update({"dbscan_eps": eps, "dbscan_min_samples": min_samples, "post_voxel_points": int(len(pts))})
    boxes = []
    raw_clusters = 0
    accepted = 0
    for cid in np.unique(labels):
        if cid < 0:
            continue
        raw_clusters += 1
        cluster = pts[labels == cid]
        if len(cluster) < min_samples:
            continue
        box = oriented_box_from_points(cluster)
        if box is None:
            continue
        if box["bbox_height"] < 8.0:
            continue
        if max(box["bbox_width"], box["bbox_length"]) < 2.0:
            continue
        accepted += 1
        boxes.append(box)
    merged = merge_boxes(boxes, "Wind Turbine")
    diag.update({
        "dbscan_clusters": raw_clusters,
        "accepted_clusters": accepted,
        "merged_clusters": int(len(merged)),
    })
    return merged


def detect_boxes_for_frame(frame_df: pd.DataFrame, point_fraction: float, seed: int) -> Tuple[List[Dict], Dict]:
    frame_df = frame_df[frame_df["distance_cm"] > 0].reset_index(drop=True)
    frame_df = sample_fraction(frame_df, point_fraction=point_fraction, seed=seed)
    if len(frame_df) == 0:
        return [], {"input_points": 0, "valid_points": 0}

    xyz = spherical_to_xyz_robust(frame_df)
    out_boxes: List[Dict] = []
    diagnostics: Dict[str, Dict] = {"frame": {"input_points": int(len(frame_df)), "valid_points": int(len(frame_df))}}

    for class_name in CLASS_ORDER:
        mask = class_mask_from_rgb(frame_df, class_name)
        pts = xyz[mask]
        diag = {"input_points": int(len(pts))}
        if class_name == "Cable":
            boxes = detect_cable_boxes(pts, point_fraction, diag)
        elif class_name in ("Antenna", "Electric Pole"):
            boxes = detect_vertical_boxes(pts, class_name, point_fraction, diag)
        else:
            boxes = detect_turbine_boxes(pts, point_fraction, diag)
        diagnostics[class_name] = diag
        class_id = CLASS_NAME_TO_ID[class_name]
        for b in boxes:
            b["class_ID"] = class_id
            b["class_label"] = class_name
            out_boxes.append(b)
    return out_boxes, diagnostics


def process_pose(payload: Tuple[int, Dict[str, float], pd.DataFrame, float, int]) -> Tuple[int, List[Dict], Dict]:
    pose_index, pose_dict, pose_df, point_fraction, seed = payload
    boxes, diag = detect_boxes_for_frame(pose_df, point_fraction=point_fraction, seed=seed + pose_index)
    rows = []
    for box in boxes:
        rows.append(
            {
                "ego_x": float(pose_dict["ego_x"]),
                "ego_y": float(pose_dict["ego_y"]),
                "ego_z": float(pose_dict["ego_z"]),
                "ego_yaw": float(pose_dict["ego_yaw"]),
                **box,
            }
        )
    return pose_index, rows, diag


def visualize_frame(*args, **kwargs):
    # Intentionally no-op on compute servers. Keep CLI compatibility.
    print("[WARN] Visualization disabled in v6 on headless/compute nodes.")
    return False


def process_file(input_h5: Path, output_csv: Path, num_workers: int, point_fraction: float, seed: int,
                 visualize_pose_index: Optional[int] = None, diagnostics_json: Optional[Path] = None) -> pd.DataFrame:
    df = load_h5_as_dataframe(input_h5)
    poses_df = get_unique_poses_df(df)

    payloads = []
    for pose_index, pose_row in poses_df.iterrows():
        frame_df = filter_by_pose_df(df, pose_row)
        payloads.append((pose_index, pose_row.to_dict(), frame_df, point_fraction, seed))

    results = []
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for res in ex.map(process_pose, payloads, chunksize=1):
                results.append(res)
    else:
        results = [process_pose(p) for p in payloads]

    results.sort(key=lambda x: x[0])
    rows: List[Dict] = []
    diag_out: Dict[str, Dict] = {}
    for pose_index, pose_rows, diag in results:
        rows.extend(pose_rows)
        diag_out[f"pose_{pose_index:03d}"] = diag
        if visualize_pose_index is not None and pose_index == visualize_pose_index:
            print("[INFO] Visualization skipped.")

    pred_df = pd.DataFrame(rows, columns=[
        "ego_x", "ego_y", "ego_z", "ego_yaw",
        "bbox_center_x", "bbox_center_y", "bbox_center_z",
        "bbox_width", "bbox_length", "bbox_height", "bbox_yaw",
        "class_ID", "class_label",
    ])
    pred_df.to_csv(output_csv, index=False)

    if diagnostics_json is not None:
        with diagnostics_json.open("w", encoding="utf-8") as f:
            json.dump(diag_out, f, indent=2)

    print(f"Saved {len(pred_df)} detections to {output_csv}")
    if len(pred_df) > 0:
        print(pred_df.head(10).to_string(index=False))
    return pred_df


def main() -> None:
    args = parse_args()
    process_file(
        input_h5=args.file,
        output_csv=args.output_csv,
        num_workers=max(1, args.num_workers),
        point_fraction=float(args.point_fraction),
        seed=int(args.seed),
        visualize_pose_index=args.visualize_pose_index,
        diagnostics_json=args.diagnostics_json,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN

import lidar_utils
from train_small_point_model import SmallPointMLP


CLASS_ID_TO_NAME = {
    1: "Antenna",
    2: "Cable",
    3: "Electric Pole",
    4: "Wind Turbine",
}

CLASS_NAME_TO_ID = {v: k for k, v in CLASS_ID_TO_NAME.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with small point model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_h5_as_dataframe(h5_path: Path) -> pd.DataFrame:
    if hasattr(lidar_utils, "load_h5_data"):
        df = lidar_utils.load_h5_data(str(h5_path))
        if isinstance(df, pd.DataFrame):
            return df

    with h5py.File(h5_path, "r") as f:
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
        (df["ego_x"] == pose_row["ego_x"])
        & (df["ego_y"] == pose_row["ego_y"])
        & (df["ego_z"] == pose_row["ego_z"])
        & (df["ego_yaw"] == pose_row["ego_yaw"])
    )
    return df.loc[mask].reset_index(drop=True)


def spherical_to_xyz_robust(frame_df: pd.DataFrame) -> np.ndarray:
    fn = lidar_utils.spherical_to_local_cartesian

    try:
        sig = inspect.signature(fn)
        n_params = len(sig.parameters)
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
    return xyz


def normalize_features(xyz: np.ndarray, reflectivity: np.ndarray) -> np.ndarray:
    xyz_centered = xyz - xyz.mean(axis=0, keepdims=True)

    refl = reflectivity.astype(np.float32).reshape(-1, 1)
    if refl.size > 0:
        mn, mx = float(refl.min()), float(refl.max())
        if mx > mn:
            refl = (refl - mn) / (mx - mn)
        else:
            refl = np.zeros_like(refl)

    feats = np.concatenate([xyz_centered.astype(np.float32), refl], axis=1)
    return feats.astype(np.float32)


def load_model(model_path: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    in_dim = ckpt.get("in_dim", 4)
    num_classes = ckpt.get("num_classes", 5)

    model = SmallPointMLP(in_dim=in_dim, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def fit_oriented_box(points: np.ndarray, class_name: str) -> Dict[str, float] | None:
    if len(points) < 5:
        return None

    center = points.mean(axis=0)
    centered = points - center

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    axis0 = eigvecs[:, 0]
    yaw = float(np.arctan2(axis0[1], axis0[0]))

    rot = np.array(
        [
            [np.cos(-yaw), -np.sin(-yaw), 0.0],
            [np.sin(-yaw),  np.cos(-yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    local = centered @ rot.T
    mins = local.min(axis=0)
    maxs = local.max(axis=0)
    extents = maxs - mins

    width = float(max(extents[0], 0.05))
    length = float(max(extents[1], 0.05))
    height = float(max(extents[2], 0.05))

    if class_name == "Cable":
        width = min(width, 0.15)
        height = min(height, 1.5)
        length = max(length, width)

    return {
        "bbox_center_x": float(center[0]),
        "bbox_center_y": float(center[1]),
        "bbox_center_z": float(center[2]),
        "bbox_width": width,
        "bbox_length": length,
        "bbox_height": height,
        "bbox_yaw": yaw,
    }


def postprocess_class_points(points: np.ndarray, class_name: str) -> List[Dict[str, float]]:
    if len(points) == 0:
        return []

    if class_name == "Cable":
        eps, min_samples = 1.2, 10
    elif class_name == "Wind Turbine":
        eps, min_samples = 2.5, 20
    else:
        eps, min_samples = 1.0, 10

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    out = []

    for cid in np.unique(labels):
        if cid < 0:
            continue
        cluster = points[labels == cid]
        if len(cluster) < min_samples:
            continue

        box = fit_oriented_box(cluster, class_name)
        if box is None:
            continue

        if class_name in ("Antenna", "Electric Pole"):
            vertical_ratio = box["bbox_height"] / max(box["bbox_width"], box["bbox_length"], 1e-3)
            if vertical_ratio < 1.5:
                continue

        if class_name == "Wind Turbine":
            if box["bbox_height"] < 10.0:
                continue

        out.append(box)

    return out


def infer_file(model, h5_path: Path, device: torch.device, score_threshold: float = 0.5) -> pd.DataFrame:
    df = load_h5_as_dataframe(h5_path)
    poses_df = get_unique_poses_df(df)

    rows = []

    for _, pose_row in poses_df.iterrows():
        frame_df = filter_by_pose_df(df, pose_row)
        frame_df = frame_df[frame_df["distance_cm"] > 0].reset_index(drop=True)
        if len(frame_df) == 0:
            continue

        xyz = spherical_to_xyz_robust(frame_df)
        reflectivity = frame_df["reflectivity"].to_numpy().astype(np.float32)
        features = normalize_features(xyz, reflectivity)

        x = torch.from_numpy(features).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)[0]
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)

        conf = conf.cpu().numpy()
        pred = pred.cpu().numpy()

        for class_id, class_name in CLASS_ID_TO_NAME.items():
            mask = (pred == class_id) & (conf >= score_threshold)
            class_points = xyz[mask]
            boxes = postprocess_class_points(class_points, class_name)

            for box in boxes:
                rows.append(
                    {
                        "ego_x": float(pose_row["ego_x"]),
                        "ego_y": float(pose_row["ego_y"]),
                        "ego_z": float(pose_row["ego_z"]),
                        "ego_yaw": float(pose_row["ego_yaw"]),
                        **box,
                        "class_ID": class_id,
                        "class_label": class_name,
                    }
                )

    columns = [
        "ego_x", "ego_y", "ego_z", "ego_yaw",
        "bbox_center_x", "bbox_center_y", "bbox_center_z",
        "bbox_width", "bbox_length", "bbox_height", "bbox_yaw",
        "class_ID", "class_label",
    ]
    return pd.DataFrame(rows, columns=columns)


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = load_model(args.model, device)
    pred_df = infer_file(model, args.file, device=device, score_threshold=args.score_threshold)
    pred_df.to_csv(args.output_csv, index=False)

    print(f"Saved {len(pred_df)} detections to {args.output_csv}")
    if len(pred_df) > 0:
        print(pred_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
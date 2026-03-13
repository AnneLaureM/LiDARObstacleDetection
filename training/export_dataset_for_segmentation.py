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

import lidar_utils


CLASS_NAME_TO_ID = {
    "Background": 0,
    "Antenna": 1,
    "Cable": 2,
    "Electric Pole": 3,
    "Wind Turbine": 4,
}

RGB_TO_CLASS_NAME = {
    (38, 23, 180): "Antenna",
    (177, 132, 47): "Cable",
    (129, 81, 97): "Electric Pole",
    (66, 132, 9): "Wind Turbine",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Airbus HDF5 train set into a point-wise segmentation dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing scene_*.h5 files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for segmentation dataset",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Maximum number of points kept per frame after downsampling/sampling",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.15,
        help="Voxel size in meters for voxel downsampling",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_h5_as_dataframe(h5_path: Path) -> pd.DataFrame:
    """
    Robust loader:
    1) tries lidar_utils.load_h5_data if present
    2) otherwise falls back to direct HDF5 reading
    """
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
    """
    Robust pose extraction. Supports either toolkit behavior or direct pandas fallback.
    """
    pose_cols = ["ego_x", "ego_y", "ego_z", "ego_yaw"]

    if hasattr(lidar_utils, "get_unique_poses"):
        poses = lidar_utils.get_unique_poses(df)
        if isinstance(poses, pd.DataFrame):
            return poses.reset_index(drop=True)
        if isinstance(poses, np.ndarray):
            return pd.DataFrame(poses, columns=pose_cols)

    return df[pose_cols].drop_duplicates().reset_index(drop=True)


def filter_by_pose_df(df: pd.DataFrame, pose_row: pd.Series) -> pd.DataFrame:
    """
    Robust pose filtering. Supports toolkit behavior or direct pandas fallback.
    """
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
    """
    Supports both signatures:
      A) spherical_to_local_cartesian(df)
      B) spherical_to_local_cartesian(distance_cm, azimuth_raw, elevation_raw)
    """
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
        # fallback by trial
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
        raise ValueError(f"Invalid XYZ shape returned by lidar_utils: {xyz.shape}")
    return xyz


def rgb_to_labels(frame_df: pd.DataFrame) -> np.ndarray:
    """
    Map RGB to class IDs. Unknown colors -> Background.
    """
    rgb = np.stack(
        [
            frame_df["r"].to_numpy().astype(np.int32),
            frame_df["g"].to_numpy().astype(np.int32),
            frame_df["b"].to_numpy().astype(np.int32),
        ],
        axis=1,
    )

    labels = np.zeros(len(frame_df), dtype=np.int64)
    for rgb_triplet, class_name in RGB_TO_CLASS_NAME.items():
        mask = np.all(rgb == np.array(rgb_triplet, dtype=np.int32), axis=1)
        labels[mask] = CLASS_NAME_TO_ID[class_name]
    return labels


def voxel_downsample(
    xyz: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple voxel downsampling by keeping one random point per voxel.
    """
    if len(xyz) == 0:
        return xyz, features, labels

    coords = np.floor(xyz / voxel_size).astype(np.int32)

    voxel_dict: Dict[Tuple[int, int, int], int] = {}
    keep_indices: List[int] = []

    for idx, c in enumerate(coords):
        key = (int(c[0]), int(c[1]), int(c[2]))
        if key not in voxel_dict:
            voxel_dict[key] = idx
            keep_indices.append(idx)

    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    return xyz[keep_indices], features[keep_indices], labels[keep_indices]


def sample_points_balanced(
    xyz: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep up to max_points with mild class balancing.
    Background can dominate heavily, so we cap it.
    """
    n = len(xyz)
    if n <= max_points:
        return xyz, features, labels

    all_idx = np.arange(n)
    fg_idx = all_idx[labels > 0]
    bg_idx = all_idx[labels == 0]

    # target: keep foreground as much as possible, then fill with background
    target_fg = min(len(fg_idx), int(max_points * 0.7))
    target_bg = max_points - target_fg

    if len(fg_idx) > 0:
        chosen_fg = rng.choice(
            fg_idx, size=target_fg, replace=(len(fg_idx) < target_fg)
        )
    else:
        chosen_fg = np.empty((0,), dtype=np.int64)

    if len(bg_idx) > 0 and target_bg > 0:
        chosen_bg = rng.choice(
            bg_idx, size=min(target_bg, len(bg_idx)), replace=False
        )
    else:
        chosen_bg = np.empty((0,), dtype=np.int64)

    chosen = np.concatenate([chosen_fg, chosen_bg], axis=0)

    # if not enough because bg was small, fill from remainder
    if len(chosen) < max_points:
        remaining = np.setdiff1d(all_idx, chosen, assume_unique=False)
        if len(remaining) > 0:
            extra = rng.choice(
                remaining,
                size=min(max_points - len(chosen), len(remaining)),
                replace=False,
            )
            chosen = np.concatenate([chosen, extra], axis=0)

    rng.shuffle(chosen)
    return xyz[chosen], features[chosen], labels[chosen]


def normalize_features(xyz: np.ndarray, reflectivity: np.ndarray) -> np.ndarray:
    """
    Create features [x, y, z, reflectivity] with simple normalization.
    """
    xyz_centered = xyz - xyz.mean(axis=0, keepdims=True)

    refl = reflectivity.astype(np.float32).reshape(-1, 1)
    if refl.size > 0:
        refl_min = float(refl.min())
        refl_max = float(refl.max())
        if refl_max > refl_min:
            refl = (refl - refl_min) / (refl_max - refl_min)
        else:
            refl = np.zeros_like(refl)

    feats = np.concatenate([xyz_centered.astype(np.float32), refl], axis=1)
    return feats.astype(np.float32)


def extract_frame_samples(
    h5_path: Path,
    voxel_size: float,
    max_points: int,
    rng: np.random.Generator,
) -> List[dict]:
    df = load_h5_as_dataframe(h5_path)
    poses_df = get_unique_poses_df(df)

    samples: List[dict] = []

    for pose_idx, pose_row in poses_df.iterrows():
        frame_df = filter_by_pose_df(df, pose_row)

        # remove invalid returns
        frame_df = frame_df[frame_df["distance_cm"] > 0].reset_index(drop=True)
        if len(frame_df) == 0:
            continue

        required_cols = {
            "distance_cm", "azimuth_raw", "elevation_raw",
            "reflectivity", "r", "g", "b",
            "ego_x", "ego_y", "ego_z", "ego_yaw",
        }
        missing = required_cols - set(frame_df.columns)
        if missing:
            raise KeyError(f"{h5_path.name}: missing columns: {sorted(missing)}")

        xyz = spherical_to_xyz_robust(frame_df)
        reflectivity = frame_df["reflectivity"].to_numpy().astype(np.float32)
        labels = rgb_to_labels(frame_df)

        features = normalize_features(xyz, reflectivity)

        xyz_ds, features_ds, labels_ds = voxel_downsample(
            xyz, features, labels, voxel_size=voxel_size
        )

        xyz_out, features_out, labels_out = sample_points_balanced(
            xyz_ds, features_ds, labels_ds, max_points=max_points, rng=rng
        )

        meta = {
            "scene_name": h5_path.stem,
            "pose_index": int(pose_idx),
            "ego_x": float(pose_row["ego_x"]),
            "ego_y": float(pose_row["ego_y"]),
            "ego_z": float(pose_row["ego_z"]),
            "ego_yaw": float(pose_row["ego_yaw"]),
            "num_points_before": int(len(frame_df)),
            "num_points_after": int(len(xyz_out)),
        }

        samples.append(
            {
                "xyz": xyz_out.astype(np.float32),
                "features": features_out.astype(np.float32),
                "labels": labels_out.astype(np.int64),
                "meta": meta,
            }
        )

    return samples


def save_sample(sample: dict, out_path: Path) -> None:
    meta_json = json.dumps(sample["meta"], ensure_ascii=False)
    np.savez_compressed(
        out_path,
        xyz=sample["xyz"],
        features=sample["features"],
        labels=sample["labels"],
        meta_json=np.array(meta_json),
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(input_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {input_dir}")

    all_samples: List[dict] = []
    per_scene_counts = {}

    for h5_path in h5_files:
        scene_samples = extract_frame_samples(
            h5_path=h5_path,
            voxel_size=args.voxel_size,
            max_points=args.max_points,
            rng=rng,
        )
        per_scene_counts[h5_path.name] = len(scene_samples)
        all_samples.extend(scene_samples)
        print(f"[OK] {h5_path.name}: extracted {len(scene_samples)} frame samples")

    if not all_samples:
        raise RuntimeError("No samples extracted from input dataset")

    indices = np.arange(len(all_samples))
    rng.shuffle(indices)

    n_val = max(1, int(len(indices) * args.val_ratio))
    val_indices = set(indices[:n_val].tolist())

    train_count = 0
    val_count = 0

    for idx, sample in enumerate(all_samples):
        scene_name = sample["meta"]["scene_name"]
        pose_index = sample["meta"]["pose_index"]
        filename = f"{scene_name}_pose{pose_index:03d}.npz"

        if idx in val_indices:
            save_sample(sample, val_dir / filename)
            val_count += 1
        else:
            save_sample(sample, train_dir / filename)
            train_count += 1

    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_scenes": len(h5_files),
        "num_samples_total": len(all_samples),
        "num_train_samples": train_count,
        "num_val_samples": val_count,
        "max_points": args.max_points,
        "voxel_size": args.voxel_size,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "class_name_to_id": CLASS_NAME_TO_ID,
        "rgb_to_class_name": {str(k): v for k, v in RGB_TO_CLASS_NAME.items()},
        "per_scene_counts": per_scene_counts,
    }

    with (output_dir / "dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nExport completed.")
    print(f"Train samples: {train_count}")
    print(f"Val samples:   {val_count}")
    print(f"Metadata:      {output_dir / 'dataset_metadata.json'}")


if __name__ == "__main__":
    main()
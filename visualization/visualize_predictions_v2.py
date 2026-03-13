#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Normalize

import lidar_utils


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

CLASS_COLORS = {
    "Antenna": (0.15, 0.25, 1.0),
    "Cable": (1.0, 0.7, 0.15),
    "Electric Pole": (0.75, 0.3, 0.65),
    "Wind Turbine": (0.1, 0.85, 0.25),
}

ALT_CLASS_COLORS = {
    0: CLASS_COLORS["Antenna"],
    1: CLASS_COLORS["Cable"],
    2: CLASS_COLORS["Electric Pole"],
    3: CLASS_COLORS["Wind Turbine"],
    4: CLASS_COLORS["Wind Turbine"],
}

POSE_FIELDS = ["ego_x", "ego_y", "ego_z", "ego_yaw"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Airbus LiDAR point cloud with predicted 3D bounding boxes."
    )
    parser.add_argument("--file", required=True, help="Path to input HDF5 file")
    parser.add_argument("--pred-csv", required=True, help="Path to predictions CSV")
    parser.add_argument(
        "--pose-index",
        type=int,
        default=None,
        help="Index of the pose to visualize (0-based). If omitted, prints available poses.",
    )
    parser.add_argument(
        "--cmap",
        default="turbo",
        help="Matplotlib colormap used when RGB is unavailable and reflectivity is shown.",
    )
    parser.add_argument(
        "--screenshot",
        default=None,
        help="Optional PNG path. If provided, captures a screenshot.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Try screenshot-only mode. Useful with xvfb-run or offscreen rendering.",
    )
    parser.add_argument(
        "--hide-point-colors",
        action="store_true",
        help="Ignore RGB/reflectivity and render points in light gray.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Open3D point size",
    )
    return parser.parse_args()


def load_predictions(csv_path: Path) -> pd.DataFrame:
    pred_df = pd.read_csv(csv_path)
    required = set(
        POSE_FIELDS
        + [
            "bbox_center_x",
            "bbox_center_y",
            "bbox_center_z",
            "bbox_width",
            "bbox_length",
            "bbox_height",
            "bbox_yaw",
            "class_ID",
            "class_label",
        ]
    )
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {sorted(missing)}")
    return pred_df


def robust_get_unique_poses(df: pd.DataFrame) -> pd.DataFrame:
    poses = lidar_utils.get_unique_poses(df)

    if isinstance(poses, pd.DataFrame):
        pose_df = poses.copy()
    else:
        pose_df = pd.DataFrame(poses, columns=POSE_FIELDS)

    pose_df = pose_df.reset_index(drop=True)

    if "pose_index" not in pose_df.columns:
        pose_df["pose_index"] = np.arange(len(pose_df), dtype=int)

    if "num_points" not in pose_df.columns:
        counts = []
        for _, row in pose_df.iterrows():
            pose_tuple = tuple(float(row[k]) for k in POSE_FIELDS)
            tmp = lidar_utils.filter_by_pose(df, pose_tuple)
            tmp = tmp[tmp["distance_cm"] > 0]
            counts.append(int(len(tmp)))
        pose_df["num_points"] = counts

    return pose_df


def robust_filter_by_pose(df: pd.DataFrame, pose_row: pd.Series) -> pd.DataFrame:
    pose_tuple = tuple(float(pose_row[k]) for k in POSE_FIELDS)
    out = lidar_utils.filter_by_pose(df, pose_tuple)
    return out.reset_index(drop=True)


def robust_spherical_to_xyz(frame_df: pd.DataFrame) -> np.ndarray:
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
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Invalid XYZ shape returned by lidar_utils: {xyz.shape}")
    return xyz


def select_pose(df: pd.DataFrame, pose_index: int | None) -> Tuple[pd.DataFrame, pd.Series]:
    pose_df = robust_get_unique_poses(df)

    if pose_index is None:
        print(
            pose_df[["pose_index", "ego_x", "ego_y", "ego_z", "ego_yaw", "num_points"]]
            .to_string(index=False, float_format="%.6f")
        )
        raise SystemExit("\nUse '--pose-index N' to visualize a specific pose.")

    if pose_index < 0 or pose_index >= len(pose_df):
        raise ValueError(f"Invalid pose index {pose_index}. File has {len(pose_df)} unique poses.")

    selected_pose = pose_df.iloc[pose_index]
    frame_df = robust_filter_by_pose(df, selected_pose)
    frame_df = frame_df[frame_df["distance_cm"] > 0].reset_index(drop=True)

    print(
        pose_df.loc[
            pose_df["pose_index"] == pose_index,
            ["pose_index", "ego_x", "ego_y", "ego_z", "ego_yaw", "num_points"],
        ].to_string(index=False, float_format="%.6f")
    )
    print(f"\nSelected pose #{pose_index} -> {len(frame_df)} valid lidar points")
    return frame_df, selected_pose


def filter_predictions_for_pose(pred_df: pd.DataFrame, pose_row: pd.Series) -> pd.DataFrame:
    mask = np.ones(len(pred_df), dtype=bool)
    for field in POSE_FIELDS:
        mask &= np.isclose(pred_df[field].astype(float).to_numpy(), float(pose_row[field]), atol=1e-6)
    return pred_df.loc[mask].reset_index(drop=True)


def build_point_cloud(frame_df: pd.DataFrame, cmap_name: str, hide_point_colors: bool) -> o3d.geometry.PointCloud:
    xyz = robust_spherical_to_xyz(frame_df)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if hide_point_colors:
        pcd.paint_uniform_color([0.75, 0.75, 0.75])
        return pcd

    if {"r", "g", "b"}.issubset(frame_df.columns):
        rgb = np.column_stack(
            (
                frame_df["r"].to_numpy() / 255.0,
                frame_df["g"].to_numpy() / 255.0,
                frame_df["b"].to_numpy() / 255.0,
            )
        )
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        print("Using dataset RGB colors for point cloud rendering.")
    elif "reflectivity" in frame_df.columns:
        intensities = frame_df["reflectivity"].to_numpy()
        norm = Normalize(vmin=float(intensities.min()), vmax=float(intensities.max()))
        cmap = colormaps.get_cmap(cmap_name)
        colors = cmap(norm(intensities))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("Using reflectivity colormap for point cloud rendering.")
    else:
        pcd.paint_uniform_color([0.8, 0.8, 0.8])
        print("No RGB or reflectivity found; using gray point cloud.")

    return pcd


def box_color_from_row(row: pd.Series):
    label = str(row["class_label"])
    if label in CLASS_COLORS:
        return CLASS_COLORS[label]
    try:
        class_id = int(row["class_ID"])
    except Exception:
        class_id = -1
    return ALT_CLASS_COLORS.get(class_id, (1.0, 0.0, 0.0))


def make_box_geometry(row: pd.Series) -> o3d.geometry.OrientedBoundingBox:
    center = np.array(
        [row["bbox_center_x"], row["bbox_center_y"], row["bbox_center_z"]],
        dtype=np.float64,
    )
    extent = np.array(
        [
            max(float(row["bbox_width"]), 1e-3),
            max(float(row["bbox_length"]), 1e-3),
            max(float(row["bbox_height"]), 1e-3),
        ],
        dtype=np.float64,
    )
    yaw = float(row["bbox_yaw"])
    R = o3d.geometry.get_rotation_matrix_from_xyz([0.0, 0.0, yaw])
    obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
    obb.color = box_color_from_row(row)
    return obb


def add_coordinate_frame(size: float = 5.0) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.0, 0.0, 0.0])


def try_capture(vis: o3d.visualization.Visualizer, screenshot_path: Path | None) -> None:
    if screenshot_path is None:
        return
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(screenshot_path))
    print(f"Saved screenshot to {screenshot_path}")


def main() -> None:
    args = parse_args()
    h5_path = Path(args.file)
    pred_csv = Path(args.pred_csv)
    screenshot_path = Path(args.screenshot) if args.screenshot else None

    df = lidar_utils.load_h5_data(str(h5_path))
    if len(df) == 0:
        raise RuntimeError("Dataset contains 0 lidar points.")

    frame_df, selected_pose = select_pose(df, args.pose_index)
    pred_df = load_predictions(pred_csv)
    pose_pred_df = filter_predictions_for_pose(pred_df, selected_pose)

    print(f"Predicted boxes for selected pose: {len(pose_pred_df)}")
    if len(pose_pred_df) > 0:
        print(
            pose_pred_df[
                [
                    "class_label",
                    "bbox_center_x",
                    "bbox_center_y",
                    "bbox_center_z",
                    "bbox_width",
                    "bbox_length",
                    "bbox_height",
                    "bbox_yaw",
                ]
            ]
            .head(20)
            .to_string(index=False, float_format="%.4f")
        )

    pcd = build_point_cloud(frame_df, args.cmap, args.hide_point_colors)
    geometries = [pcd, add_coordinate_frame(size=5.0)]

    for _, row in pose_pred_df.iterrows():
        geometries.append(make_box_geometry(row))

    pts = np.asarray(pcd.points)
    print(
        f"Bounds X:[{pts[:,0].min():.1f}, {pts[:,0].max():.1f}] "
        f"Y:[{pts[:,1].min():.1f}, {pts[:,1].max():.1f}] "
        f"Z:[{pts[:,2].min():.1f}, {pts[:,2].max():.1f}]"
    )

    title = f"{h5_path.name} pose {int(selected_pose['pose_index'])} with predictions"
    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(
        window_name=title,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        visible=not args.headless,
    )
    if not ok:
        raise RuntimeError(
            "Open3D failed to create a window. On a headless server, run this on a machine "
            "with a display, use xvfb-run, or install a working offscreen backend."
        )

    try:
        for geom in geometries:
            vis.add_geometry(geom)

        ctrl = vis.get_view_control()
        if ctrl is not None:
            ctrl.set_lookat(np.array([10.0, 0.0, 0.0], dtype=np.float64))
            ctrl.set_front(np.array([-1.0, 0.0, 0.0], dtype=np.float64))
            ctrl.set_up(np.array([0.0, 0.0, 1.0], dtype=np.float64))
            ctrl.set_zoom(0.10)

        render_opt = vis.get_render_option()
        if render_opt is not None:
            render_opt.point_size = float(args.point_size)
            render_opt.background_color = np.array([0.02, 0.02, 0.02], dtype=np.float64)

        try_capture(vis, screenshot_path)

        if not args.headless:
            vis.run()
    finally:
        vis.destroy_window()


if __name__ == "__main__":
    main()

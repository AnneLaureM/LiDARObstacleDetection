#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import open3d as o3d
except Exception:
    o3d = None

from sklearn.cluster import DBSCAN

import lidar_utils

POSE_FIELDS = ["ego_x", "ego_y", "ego_z", "ego_yaw"]
REQUIRED_COLUMNS = [
    "ego_x",
    "ego_y",
    "ego_z",
    "ego_yaw",
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

RGB_TO_CLASS = {
    (38, 23, 180): (0, "Antenna"),
    (177, 132, 47): (1, "Cable"),
    (129, 81, 97): (2, "Electric Pole"),
    (66, 132, 9): (3, "Wind Turbine"),
}
CLASS_TO_ID = {v[1]: v[0] for v in RGB_TO_CLASS.values()}

CLASS_CFG = {
    "Antenna": {
        "dbscan_eps": 1.0,
        "dbscan_min_samples": 10,
        "min_points": 15,
        "min_height": 1.5,
        "max_xy": 5.5,
        "vertical_ratio": 1.8,
        "voxel_size": 0.18,
        "max_points_after_voxel": 25000,
    },
    "Cable": {
        "dbscan_eps": 1.6,
        "dbscan_min_samples": 8,
        "min_points": 8,
        "min_length": 4.0,
        "max_thickness": 1.35,
        "linearity_min": 6.0,
        "merge_axis_cos": 0.955,
        "merge_endpoint_dist": 8.0,
        "merge_perp_dist": 2.0,
        "voxel_size": 0.12,
        "max_points_after_voxel": 90000,
    },
    "Electric Pole": {
        "dbscan_eps": 1.35,
        "dbscan_min_samples": 10,
        "min_points": 18,
        "min_height": 4.0,
        "max_xy": 3.2,
        "vertical_ratio": 2.3,
        "voxel_size": 0.16,
        "max_points_after_voxel": 30000,
    },
    "Wind Turbine": {
        "dbscan_eps": 2.8,
        "dbscan_min_samples": 16,
        "min_points": 40,
        "min_height": 8.0,
        "min_xy": 0.8,
        "merge_center_dist": 14.0,
        "voxel_size": 0.35,
        "max_points_after_voxel": 50000,
    },
}


@dataclass
class Box3D:
    ego_x: float
    ego_y: float
    ego_z: float
    ego_yaw: float
    bbox_center_x: float
    bbox_center_y: float
    bbox_center_z: float
    bbox_width: float
    bbox_length: float
    bbox_height: float
    bbox_yaw: float
    class_ID: int
    class_label: str

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "ego_x": self.ego_x,
            "ego_y": self.ego_y,
            "ego_z": self.ego_z,
            "ego_yaw": self.ego_yaw,
            "bbox_center_x": self.bbox_center_x,
            "bbox_center_y": self.bbox_center_y,
            "bbox_center_z": self.bbox_center_z,
            "bbox_width": self.bbox_width,
            "bbox_length": self.bbox_length,
            "bbox_height": self.bbox_height,
            "bbox_yaw": self.bbox_yaw,
            "class_ID": self.class_ID,
            "class_label": self.class_label,
        }


@dataclass
class ClusterGeom:
    label: str
    points: np.ndarray
    center: np.ndarray
    extents: np.ndarray
    yaw: float
    axes: np.ndarray
    evals: np.ndarray
    min_corner: np.ndarray
    max_corner: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline LiDAR detector v5: CPU-optimized multiprocessing + downsampling")
    parser.add_argument("--file", type=Path, required=True, help="Input .h5 file")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--visualize-pose-index", type=int, default=None, help="Pose index to visualize")
    parser.add_argument("--point-fraction", type=float, default=1.0, help="Optional random point fraction in (0,1]")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for subsampling")
    parser.add_argument("--max-poses", type=int, default=None, help="Optional cap on number of poses to process")
    parser.add_argument("--num-workers", type=int, default=max(1, (os_cpu_count() or 2) - 1), help="CPU workers for per-pose processing")
    parser.add_argument("--disable-voxel-downsampling", action="store_true", help="Disable per-class voxel downsampling")
    parser.add_argument("--dbscan-scale", type=float, default=1.0, help="Global multiplier applied to DBSCAN eps")
    parser.add_argument("--diagnostics-json", type=Path, default=None, help="Optional JSON output with clustering diagnostics")
    return parser.parse_args()


def os_cpu_count() -> int | None:
    try:
        return mp.cpu_count()
    except Exception:
        return None


def _local_xyz_from_df(df: pd.DataFrame) -> np.ndarray:
    sig = inspect.signature(lidar_utils.spherical_to_local_cartesian)
    if len(sig.parameters) == 1:
        return np.asarray(lidar_utils.spherical_to_local_cartesian(df), dtype=np.float32)
    return np.asarray(
        lidar_utils.spherical_to_local_cartesian(
            df["distance_cm"].to_numpy(),
            df["azimuth_raw"].to_numpy(),
            df["elevation_raw"].to_numpy(),
        ),
        dtype=np.float32,
    )


def subsample_frame_df(frame_df: pd.DataFrame, point_fraction: float, seed: int, pose_index: int) -> pd.DataFrame:
    if point_fraction >= 1.0:
        return frame_df.reset_index(drop=True)
    if point_fraction <= 0.0:
        raise ValueError("point_fraction must be > 0")
    rng = np.random.default_rng(seed + pose_index)
    n = len(frame_df)
    k = max(1, int(np.floor(n * point_fraction)))
    idx = np.sort(rng.choice(n, size=k, replace=False))
    return frame_df.iloc[idx].reset_index(drop=True)


def split_into_frames(df: pd.DataFrame) -> list[pd.DataFrame]:
    poses = lidar_utils.get_unique_poses(df)
    if poses is None or len(poses) == 0:
        return [df.reset_index(drop=True)]
    if isinstance(poses, np.ndarray):
        poses = pd.DataFrame(poses, columns=POSE_FIELDS)
    frames: list[pd.DataFrame] = []
    for _, pose_row in poses.iterrows():
        frames.append(lidar_utils.filter_by_pose(df, pose_row).reset_index(drop=True))
    return frames


def dataframe_to_local_xyz(frame_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    valid_df = frame_df[frame_df["distance_cm"] > 0].reset_index(drop=True)
    if len(valid_df) == 0:
        return valid_df, np.zeros((0, 3), dtype=np.float32)
    xyz = _local_xyz_from_df(valid_df)
    return valid_df, xyz


def rgb_to_class_mask(df: pd.DataFrame, label: str) -> np.ndarray:
    wanted = None
    for rgb, (_, lbl) in RGB_TO_CLASS.items():
        if lbl == label:
            wanted = rgb
            break
    if wanted is None:
        raise KeyError(label)
    r, g, b = wanted
    return (
        (df["r"].to_numpy() == r)
        & (df["g"].to_numpy() == g)
        & (df["b"].to_numpy() == b)
    )


def voxel_downsample(points: np.ndarray, voxel_size: float, max_points: int | None = None) -> np.ndarray:
    if len(points) == 0 or voxel_size <= 0:
        out = points
    else:
        q = np.floor(points / voxel_size).astype(np.int32)
        _, uniq_idx = np.unique(q, axis=0, return_index=True)
        out = points[np.sort(uniq_idx)]
    if max_points is not None and len(out) > max_points:
        step = max(1, len(out) // max_points)
        out = out[::step][:max_points]
    return out


def pca_geometry(points: np.ndarray, label: str) -> ClusterGeom:
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T) if len(points) > 1 else np.eye(3, dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    proj = centered @ evecs
    min_proj = proj.min(axis=0)
    max_proj = proj.max(axis=0)
    extents = np.maximum(max_proj - min_proj, 1e-3)
    mid_proj = (max_proj + min_proj) * 0.5
    box_center = center + evecs @ mid_proj
    yaw = float(np.arctan2(evecs[1, 0], evecs[0, 0]))
    return ClusterGeom(
        label=label,
        points=points,
        center=box_center,
        extents=extents,
        yaw=yaw,
        axes=evecs,
        evals=evals,
        min_corner=points.min(axis=0),
        max_corner=points.max(axis=0),
    )


def dbscan_clusters(points: np.ndarray, eps: float, min_samples: int) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1).fit_predict(points)
    clusters = []
    for cid in np.unique(labels):
        if cid < 0:
            continue
        clusters.append(points[labels == cid])
    return clusters


def linearity_score(evals: np.ndarray) -> float:
    if evals[1] <= 1e-6:
        return float("inf")
    return float((evals[0] + 1e-6) / (evals[1] + 1e-6))


def vertical_ratio(geom: ClusterGeom) -> float:
    xy = max(float(min(geom.extents[0], geom.extents[1])), 1e-3)
    h = float(geom.extents[2])
    return h / xy


def cable_segment_endpoints(geom: ClusterGeom) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = geom.axes[:, 0]
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    t = (geom.points - geom.center) @ axis
    p0 = geom.center + axis * t.min()
    p1 = geom.center + axis * t.max()
    return p0, p1, axis


def point_line_distance(p: np.ndarray, a: np.ndarray, u: np.ndarray) -> float:
    d = p - a
    return float(np.linalg.norm(d - np.dot(d, u) * u))


def should_merge_cables(a: ClusterGeom, b: ClusterGeom, cfg: dict) -> bool:
    a0, a1, ua = cable_segment_endpoints(a)
    b0, b1, ub = cable_segment_endpoints(b)
    cosang = abs(float(np.dot(ua, ub)))
    if cosang < cfg["merge_axis_cos"]:
        return False
    endpoint_dist = min(
        np.linalg.norm(a0 - b0), np.linalg.norm(a0 - b1), np.linalg.norm(a1 - b0), np.linalg.norm(a1 - b1)
    )
    if endpoint_dist > cfg["merge_endpoint_dist"]:
        return False
    perp = min(point_line_distance(b.center, a0, ua), point_line_distance(a.center, b0, ub))
    return perp <= cfg["merge_perp_dist"]


def merge_cluster_points(geoms: list[ClusterGeom], label: str) -> list[np.ndarray]:
    if not geoms:
        return []
    if label != "Cable":
        return [g.points for g in geoms]

    cfg = CLASS_CFG["Cable"]
    n = len(geoms)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if should_merge_cables(geoms[i], geoms[j], cfg):
                union(i, j)

    groups: dict[int, list[np.ndarray]] = {}
    for i, g in enumerate(geoms):
        groups.setdefault(find(i), []).append(g.points)
    return [np.concatenate(v, axis=0) for v in groups.values()]


def should_merge_turbines(a: ClusterGeom, b: ClusterGeom, cfg: dict) -> bool:
    z_overlap = min(a.max_corner[2], b.max_corner[2]) - max(a.min_corner[2], b.min_corner[2])
    close_in_xy = float(np.linalg.norm(a.center[:2] - b.center[:2])) <= cfg["merge_center_dist"]
    return close_in_xy and (z_overlap > -5.0 or float(np.linalg.norm(a.center - b.center)) <= cfg["merge_center_dist"])


def merge_turbine_geoms(geoms: list[ClusterGeom]) -> list[np.ndarray]:
    if not geoms:
        return []
    cfg = CLASS_CFG["Wind Turbine"]
    n = len(geoms)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if should_merge_turbines(geoms[i], geoms[j], cfg):
                union(i, j)

    groups: dict[int, list[np.ndarray]] = {}
    for i, g in enumerate(geoms):
        groups.setdefault(find(i), []).append(g.points)
    return [np.concatenate(v, axis=0) for v in groups.values()]


def class_cfg_for_runtime(label: str, point_fraction: float, dbscan_scale: float) -> dict:
    cfg = dict(CLASS_CFG[label])
    # Slightly relax eps when density drops.
    density_boost = 1.0 / math.sqrt(max(point_fraction, 1e-3))
    density_boost = min(max(density_boost, 1.0), 1.8)
    cfg["dbscan_eps"] *= dbscan_scale * density_boost
    if point_fraction < 1.0:
        cfg["dbscan_min_samples"] = max(4, int(round(cfg["dbscan_min_samples"] * math.sqrt(point_fraction))))
    return cfg


def cluster_points_for_label(points: np.ndarray, label: str, point_fraction: float, dbscan_scale: float, use_voxel: bool) -> tuple[list[np.ndarray], dict]:
    cfg = class_cfg_for_runtime(label, point_fraction=point_fraction, dbscan_scale=dbscan_scale)
    original_points = len(points)
    if use_voxel:
        points = voxel_downsample(points, cfg.get("voxel_size", 0.0), cfg.get("max_points_after_voxel"))
    voxel_points = len(points)
    clusters = dbscan_clusters(points, eps=cfg["dbscan_eps"], min_samples=cfg["dbscan_min_samples"])
    geoms: list[ClusterGeom] = []
    rejected = 0

    for cluster in clusters:
        if len(cluster) < cfg.get("min_points", 1):
            rejected += 1
            continue
        geom = pca_geometry(cluster, label)
        if label == "Cable":
            length = float(max(geom.extents))
            thickness = float(np.sort(geom.extents)[1])
            if length < cfg["min_length"] or thickness > cfg["max_thickness"] or linearity_score(geom.evals) < cfg["linearity_min"]:
                rejected += 1
                continue
        elif label in {"Antenna", "Electric Pole"}:
            if float(geom.extents[2]) < cfg["min_height"] or float(max(geom.extents[0], geom.extents[1])) > cfg["max_xy"] or vertical_ratio(geom) < cfg["vertical_ratio"]:
                rejected += 1
                continue
        elif label == "Wind Turbine":
            if float(geom.extents[2]) < cfg["min_height"] or float(max(geom.extents[0], geom.extents[1])) < cfg["min_xy"]:
                rejected += 1
                continue
        geoms.append(geom)

    if label == "Cable":
        merged = merge_cluster_points(geoms, label)
    elif label == "Wind Turbine":
        merged = merge_turbine_geoms(geoms)
    else:
        merged = [g.points for g in geoms]

    diag = {
        "label": label,
        "input_points": original_points,
        "post_voxel_points": voxel_points,
        "dbscan_clusters": len(clusters),
        "accepted_clusters": len(geoms),
        "merged_clusters": len(merged),
        "rejected_clusters": rejected,
        "dbscan_eps": cfg["dbscan_eps"],
        "dbscan_min_samples": cfg["dbscan_min_samples"],
    }
    return merged, diag


def make_box_from_cluster(points: np.ndarray, label: str, pose_meta: dict[str, float]) -> Box3D | None:
    if len(points) == 0:
        return None
    geom = pca_geometry(points, label)
    if label == "Cable":
        axis = geom.axes[:, 0]
        axis = axis / max(np.linalg.norm(axis), 1e-8)
        t = (points - geom.center) @ axis
        length = max(float(t.max() - t.min()), 0.5)
        residual = points - (geom.center + np.outer(t, axis))
        width = max(float(np.percentile(np.linalg.norm(residual[:, :2], axis=1), 95) * 2.0), 0.05)
        zmin, zmax = float(points[:, 2].min()), float(points[:, 2].max())
        height = max(zmax - zmin, 0.05)
        width = min(width, 1.0)
        height = min(max(height, 0.08), 1.5)
        yaw = float(np.arctan2(axis[1], axis[0]))
        center = np.array([
            (points[:, 0].min() + points[:, 0].max()) / 2.0,
            (points[:, 1].min() + points[:, 1].max()) / 2.0,
            (zmin + zmax) / 2.0,
        ], dtype=np.float64)
        extents = np.array([width, length, height], dtype=np.float64)
    elif label in {"Antenna", "Electric Pole"}:
        xy_center = points[:, :2].mean(axis=0)
        zmin, zmax = float(points[:, 2].min()), float(points[:, 2].max())
        xspan = float(points[:, 0].max() - points[:, 0].min())
        yspan = float(points[:, 1].max() - points[:, 1].min())
        width = max(min(xspan, yspan), 0.1)
        length = max(max(xspan, yspan), 0.1)
        height = max(zmax - zmin, 0.3)
        yaw = float(np.arctan2(geom.axes[1, 0], geom.axes[0, 0]))
        center = np.array([xy_center[0], xy_center[1], (zmin + zmax) / 2.0], dtype=np.float64)
        extents = np.array([width, length, height], dtype=np.float64)
    else:
        xy = points[:, :2]
        xy_center = xy.mean(axis=0)
        xy_cov = np.cov((xy - xy_center).T) if len(points) > 2 else np.eye(2)
        evals, evecs = np.linalg.eigh(xy_cov)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        proj_xy = (xy - xy_center) @ evecs
        min_xy = proj_xy.min(axis=0)
        max_xy = proj_xy.max(axis=0)
        width = max(float(max_xy[1] - min_xy[1]), 0.5)
        length = max(float(max_xy[0] - min_xy[0]), 0.5)
        zmin, zmax = float(points[:, 2].min()), float(points[:, 2].max())
        height = max(zmax - zmin, 1.0)
        yaw = float(np.arctan2(evecs[1, 0], evecs[0, 0]))
        center_xy_local = xy_center + evecs @ ((min_xy + max_xy) * 0.5)
        center = np.array([center_xy_local[0], center_xy_local[1], (zmin + zmax) / 2.0], dtype=np.float64)
        extents = np.array([width, length, height], dtype=np.float64)

    return Box3D(
        ego_x=float(pose_meta["ego_x"]),
        ego_y=float(pose_meta["ego_y"]),
        ego_z=float(pose_meta["ego_z"]),
        ego_yaw=float(pose_meta["ego_yaw"]),
        bbox_center_x=float(center[0]),
        bbox_center_y=float(center[1]),
        bbox_center_z=float(center[2]),
        bbox_width=float(extents[0]),
        bbox_length=float(extents[1]),
        bbox_height=float(extents[2]),
        bbox_yaw=float(yaw),
        class_ID=int(CLASS_TO_ID[label]),
        class_label=label,
    )


def deduplicate_boxes(boxes: list[Box3D]) -> list[Box3D]:
    if not boxes:
        return []
    kept: list[Box3D] = []
    for box in sorted(boxes, key=lambda b: (b.class_ID, -b.bbox_height * b.bbox_length)):
        same = False
        for prev in kept:
            if prev.class_label != box.class_label:
                continue
            c1 = np.array([prev.bbox_center_x, prev.bbox_center_y, prev.bbox_center_z])
            c2 = np.array([box.bbox_center_x, box.bbox_center_y, box.bbox_center_z])
            dist = np.linalg.norm(c1 - c2)
            size1 = np.array([prev.bbox_width, prev.bbox_length, prev.bbox_height])
            size2 = np.array([box.bbox_width, box.bbox_length, box.bbox_height])
            if dist < 0.5 * (np.linalg.norm(size1) + np.linalg.norm(size2)) / 3.0:
                same = True
                break
        if not same:
            kept.append(box)
    return kept


def detect_boxes_for_frame(frame_df: pd.DataFrame, point_fraction: float, dbscan_scale: float, use_voxel: bool) -> tuple[list[Box3D], dict]:
    frame_df, xyz = dataframe_to_local_xyz(frame_df)
    if len(frame_df) == 0 or len(xyz) == 0:
        return [], {"classes": {}, "n_valid_points": 0}

    pose_meta = {k: frame_df.iloc[0][k] for k in POSE_FIELDS}
    boxes: list[Box3D] = []
    classes_diag: dict[str, dict] = {}

    for label in ["Antenna", "Cable", "Electric Pole", "Wind Turbine"]:
        mask = rgb_to_class_mask(frame_df, label)
        cls_points = xyz[mask]
        if len(cls_points) == 0:
            classes_diag[label] = {"label": label, "input_points": 0, "post_voxel_points": 0, "dbscan_clusters": 0, "accepted_clusters": 0, "merged_clusters": 0, "rejected_clusters": 0}
            continue
        merged_clusters, diag = cluster_points_for_label(cls_points, label, point_fraction=point_fraction, dbscan_scale=dbscan_scale, use_voxel=use_voxel)
        classes_diag[label] = diag
        for cluster in merged_clusters:
            box = make_box_from_cluster(cluster, label, pose_meta)
            if box is not None:
                boxes.append(box)
    final_boxes = deduplicate_boxes(boxes)
    return final_boxes, {"classes": classes_diag, "n_valid_points": int(len(xyz)), "n_boxes": int(len(final_boxes))}


def boxes_to_dataframe(boxes: Iterable[Box3D]) -> pd.DataFrame:
    rows = [b.as_dict() for b in boxes]
    if not rows:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    df = pd.DataFrame(rows)
    return df[REQUIRED_COLUMNS]


def box_to_o3d(box: Box3D):
    if o3d is None:
        return None
    center = np.array([box.bbox_center_x, box.bbox_center_y, box.bbox_center_z], dtype=np.float64)
    extent = np.array([box.bbox_width, box.bbox_length, box.bbox_height], dtype=np.float64)
    R = o3d.geometry.get_rotation_matrix_from_xyz([0.0, 0.0, box.bbox_yaw])
    obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
    color_map = {
        "Antenna": [0.2, 0.2, 1.0],
        "Cable": [1.0, 0.7, 0.2],
        "Electric Pole": [0.8, 0.3, 0.6],
        "Wind Turbine": [0.2, 0.8, 0.2],
    }
    obb.color = color_map.get(box.class_label, [1.0, 0.0, 0.0])
    return obb


def visualize_frame(frame_df: pd.DataFrame, boxes: list[Box3D], title: str = "frame") -> bool:
    if o3d is None:
        print("[WARN] Open3D not available; skipping visualization.")
        return False
    frame_df, xyz = dataframe_to_local_xyz(frame_df)
    if len(xyz) == 0:
        print("[WARN] No valid points to visualize.")
        return False
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if {"r", "g", "b"}.issubset(frame_df.columns):
        rgb = np.column_stack([frame_df["r"].to_numpy() / 255.0, frame_df["g"].to_numpy() / 255.0, frame_df["b"].to_numpy() / 255.0])
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    else:
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(window_name=title, width=1280, height=720, visible=True)
    if not ok:
        print("[WARN] Open3D window creation failed. Headless environment detected; skipping visualization.")
        return False
    try:
        vis.add_geometry(pcd)
        for box in boxes:
            obb = box_to_o3d(box)
            if obb is not None:
                vis.add_geometry(obb)
        ctrl = vis.get_view_control()
        if ctrl is not None:
            ctrl.set_lookat(np.array([10.0, 0.0, 0.0], dtype=np.float64))
            ctrl.set_front(np.array([-1.0, 0.0, 0.0], dtype=np.float64))
            ctrl.set_up(np.array([0.0, 0.0, 1.0], dtype=np.float64))
            ctrl.set_zoom(0.12)
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        return True
    finally:
        vis.destroy_window()


def _process_pose_worker(args: tuple[int, pd.DataFrame, float, int, float, bool]) -> tuple[int, list[dict], dict]:
    pose_index, frame_df, point_fraction, seed, dbscan_scale, use_voxel = args
    frame_df = subsample_frame_df(frame_df, point_fraction=point_fraction, seed=seed, pose_index=pose_index)
    boxes, diag = detect_boxes_for_frame(frame_df, point_fraction=point_fraction, dbscan_scale=dbscan_scale, use_voxel=use_voxel)
    return pose_index, [b.as_dict() for b in boxes], diag


def process_file(input_h5: Path, output_csv: Path, point_fraction: float = 1.0, seed: int = 42,
                 visualize_pose_index: int | None = None, max_poses: int | None = None,
                 num_workers: int = 1, dbscan_scale: float = 1.0, diagnostics_json: Path | None = None,
                 disable_voxel_downsampling: bool = False) -> pd.DataFrame:
    df = lidar_utils.load_h5_data(str(input_h5)) if hasattr(lidar_utils, "load_h5_data") else lidar_utils.load_h5_data(input_h5)
    frames = split_into_frames(df)
    if max_poses is not None:
        frames = frames[:max_poses]

    tasks = [(i, frame, point_fraction, seed, dbscan_scale, not disable_voxel_downsampling) for i, frame in enumerate(frames)]

    results: list[tuple[int, list[dict], dict]] = []
    if num_workers <= 1 or len(tasks) <= 1:
        results = [_process_pose_worker(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
            results = list(ex.map(_process_pose_worker, tasks, chunksize=1))

    results.sort(key=lambda x: x[0])
    all_rows: list[dict] = []
    diags: list[dict] = []
    for pose_index, rows, diag in results:
        all_rows.extend(rows)
        diag["pose_index"] = pose_index
        diags.append(diag)

    summary_df = pd.DataFrame(all_rows, columns=REQUIRED_COLUMNS) if all_rows else pd.DataFrame(columns=REQUIRED_COLUMNS)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)

    if diagnostics_json is not None:
        diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_h5": str(input_h5),
            "point_fraction": point_fraction,
            "num_workers": num_workers,
            "dbscan_scale": dbscan_scale,
            "disable_voxel_downsampling": disable_voxel_downsampling,
            "n_poses": len(frames),
            "n_detections": int(len(summary_df)),
            "poses": diags,
        }
        diagnostics_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if visualize_pose_index is not None and 0 <= visualize_pose_index < len(frames):
        rows = [r for pose_idx, pose_rows, _ in results if pose_idx == visualize_pose_index for r in pose_rows]
        boxes = [Box3D(**r) for r in rows]
        shown = visualize_frame(frames[visualize_pose_index], boxes, title=f"{input_h5.name} pose {visualize_pose_index}")
        if not shown:
            print("[INFO] Visualization skipped.")

    return summary_df


def main() -> None:
    args = parse_args()
    summary_df = process_file(
        input_h5=args.file,
        output_csv=args.output_csv,
        point_fraction=args.point_fraction,
        seed=args.seed,
        visualize_pose_index=args.visualize_pose_index,
        max_poses=args.max_poses,
        num_workers=args.num_workers,
        dbscan_scale=args.dbscan_scale,
        diagnostics_json=args.diagnostics_json,
        disable_voxel_downsampling=args.disable_voxel_downsampling,
    )
    print(f"Saved {len(summary_df)} detections to {args.output_csv}")
    if not summary_df.empty:
        print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

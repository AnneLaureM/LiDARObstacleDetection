#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

DATASET_NAME = "lidar_points"
POSE_FIELDS = ["ego_x", "ego_y", "ego_z", "ego_yaw"]
EXPECTED_CLASSES = ["Antenna", "Cable", "Electric Pole", "Wind Turbine"]
EXPECTED_CLASS_IDS = {"Antenna": 0, "Cable": 1, "Electric Pole": 2, "Wind Turbine": 3}
EXPECTED_COLUMNS = [
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


@dataclass
class RunResult:
    scene: str
    density_pct: int
    input_h5: Path
    detector_input_h5: Path
    output_csv: Path
    n_input_points: int
    n_valid_points: int
    n_frames: int
    n_detections: int
    status: str
    stderr: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline detector across all train HDF5 files and summarize robustness at multiple point densities."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing training .h5 files",
    )
    parser.add_argument(
        "--detector-script",
        type=Path,
        default=Path("train_and_infer_baseline_v3.py"),
        help="Path to the detector/inference script to execute for each file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("batch_eval_train"),
        help="Output directory for per-scene CSVs and aggregate reports",
    )
    parser.add_argument(
        "--densities",
        type=int,
        nargs="+",
        default=[100, 75, 50, 25],
        help="Point densities to evaluate, as percentages",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for downsampling",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DATASET_NAME,
        help="Name of the HDF5 dataset containing the structured point cloud array",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch the detector script",
    )
    parser.add_argument(
        "--keep-temp-h5",
        action="store_true",
        help="Keep temporary downsampled HDF5 files for inspection",
    )
    return parser.parse_args()


def list_h5_files(input_dir: Path) -> list[Path]:
    h5_files = sorted(input_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {input_dir}")
    return h5_files


def read_structured_points(h5_path: Path, dataset_name: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if dataset_name not in f:
            raise KeyError(f"Dataset '{dataset_name}' not found in {h5_path}")
        arr = f[dataset_name][:]
    return arr


def count_frames(arr: np.ndarray) -> int:
    names = set(arr.dtype.names or [])
    if not set(POSE_FIELDS).issubset(names):
        return 1
    pose_df = pd.DataFrame({k: arr[k] for k in POSE_FIELDS})
    return int(pose_df.drop_duplicates().shape[0])


def downsample_structured_by_pose(arr: np.ndarray, density_pct: int, seed: int) -> np.ndarray:
    if density_pct >= 100:
        return arr
    if density_pct <= 0:
        raise ValueError("density_pct must be > 0")

    names = set(arr.dtype.names or [])
    if not set(POSE_FIELDS).issubset(names):
        rng = np.random.default_rng(seed)
        n_keep = max(1, int(np.floor(len(arr) * density_pct / 100.0)))
        idx = np.sort(rng.choice(len(arr), size=n_keep, replace=False))
        return arr[idx]

    df_idx = pd.DataFrame({k: arr[k] for k in POSE_FIELDS})
    grouped = df_idx.groupby(POSE_FIELDS, sort=False).indices
    rng = np.random.default_rng(seed)
    kept_indices: list[np.ndarray] = []

    for _, indices in grouped.items():
        indices = np.asarray(indices, dtype=np.int64)
        n_keep = max(1, int(np.floor(len(indices) * density_pct / 100.0)))
        chosen = np.sort(rng.choice(indices, size=n_keep, replace=False))
        kept_indices.append(chosen)

    if not kept_indices:
        return arr[:0]

    final_idx = np.concatenate(kept_indices)
    final_idx.sort()
    return arr[final_idx]


def write_structured_h5(h5_path: Path, arr: np.ndarray, dataset_name: str) -> None:
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(dataset_name, data=arr, compression="gzip")


def validate_prediction_csv(csv_path: Path) -> tuple[bool, str, pd.DataFrame]:
    if not csv_path.exists():
        return False, f"Missing prediction CSV: {csv_path}", pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return False, f"Unable to read CSV {csv_path}: {exc}", pd.DataFrame()

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"CSV {csv_path.name} missing required columns: {missing}", df

    bad_rows = df[~df["class_label"].isin(EXPECTED_CLASSES)] if not df.empty else df
    if not bad_rows.empty:
        return False, f"CSV {csv_path.name} contains unexpected class labels", df

    for label, class_id in EXPECTED_CLASS_IDS.items():
        mismatch = df[(df["class_label"] == label) & (df["class_ID"] != class_id)]
        if not mismatch.empty:
            return False, f"CSV {csv_path.name} has class_ID/class_label mismatch for {label}", df

    return True, "ok", df


def run_detector(python_exe: str, detector_script: Path, input_h5: Path, output_csv: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        python_exe,
        str(detector_script),
        "--file",
        str(input_h5),
        "--output-csv",
        str(output_csv),
    ]
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def summarize_detections(csv_df: pd.DataFrame) -> tuple[int, dict[str, int], pd.DataFrame]:
    if csv_df.empty:
        frame_counts = pd.DataFrame(columns=POSE_FIELDS + ["n_detections"])
        return 0, {cls: 0 for cls in EXPECTED_CLASSES}, frame_counts

    n_detections = int(len(csv_df))
    by_class = {cls: int((csv_df["class_label"] == cls).sum()) for cls in EXPECTED_CLASSES}
    frame_counts = (
        csv_df.groupby(POSE_FIELDS, dropna=False)
        .size()
        .reset_index(name="n_detections")
        .sort_values(POSE_FIELDS)
        .reset_index(drop=True)
    )
    return n_detections, by_class, frame_counts


def run_one_file(
    source_h5: Path,
    density_pct: int,
    dataset_name: str,
    detector_script: Path,
    python_exe: str,
    output_dir: Path,
    seed: int,
    temp_root: Path,
    keep_temp_h5: bool,
) -> tuple[RunResult, dict[str, int], pd.DataFrame]:
    arr = read_structured_points(source_h5, dataset_name)
    n_input_points = int(len(arr))
    n_valid_points = int(np.count_nonzero(arr["distance_cm"] > 0)) if "distance_cm" in (arr.dtype.names or []) else n_input_points
    n_frames = count_frames(arr)
    scene = source_h5.stem

    if density_pct == 100:
        detector_input_h5 = source_h5
    else:
        sampled = downsample_structured_by_pose(arr, density_pct=density_pct, seed=seed)
        detector_input_h5 = temp_root / f"{scene}_{density_pct}.h5"
        write_structured_h5(detector_input_h5, sampled, dataset_name=dataset_name)

    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    output_csv = pred_dir / f"{scene}_{density_pct}.csv"

    proc = run_detector(
        python_exe=python_exe,
        detector_script=detector_script,
        input_h5=detector_input_h5,
        output_csv=output_csv,
    )

    status = "ok" if proc.returncode == 0 else f"detector_failed({proc.returncode})"
    stderr = (proc.stderr or "").strip()

    by_class = {cls: 0 for cls in EXPECTED_CLASSES}
    frame_counts = pd.DataFrame(columns=POSE_FIELDS + ["n_detections"])
    n_detections = 0

    if proc.returncode == 0:
        valid, reason, pred_df = validate_prediction_csv(output_csv)
        if not valid:
            status = f"invalid_csv: {reason}"
        else:
            n_detections, by_class, frame_counts = summarize_detections(pred_df)
            frame_counts.insert(0, "density_pct", density_pct)
            frame_counts.insert(0, "scene", scene)

    if density_pct != 100 and not keep_temp_h5 and detector_input_h5.exists():
        detector_input_h5.unlink()

    result = RunResult(
        scene=scene,
        density_pct=density_pct,
        input_h5=source_h5,
        detector_input_h5=detector_input_h5,
        output_csv=output_csv,
        n_input_points=n_input_points,
        n_valid_points=n_valid_points,
        n_frames=n_frames,
        n_detections=n_detections,
        status=status,
        stderr=stderr,
    )
    return result, by_class, frame_counts


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    densities = sorted(set(args.densities), reverse=True)
    if any(d <= 0 or d > 100 for d in densities):
        raise ValueError("All densities must be in the range 1..100")

    h5_files = list_h5_files(args.input_dir)

    temp_root_obj: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_temp_h5:
        temp_root = args.output_dir / "temp_h5"
        temp_root.mkdir(parents=True, exist_ok=True)
    else:
        temp_root_obj = tempfile.TemporaryDirectory(prefix="airbus_batch_eval_")
        temp_root = Path(temp_root_obj.name)

    print(f"Found {len(h5_files)} HDF5 files in {args.input_dir}")
    print(f"Evaluating densities: {densities}")
    print(f"Detector script: {args.detector_script}")

    run_rows: list[dict] = []
    class_rows: list[dict] = []
    frame_summaries: list[pd.DataFrame] = []

    try:
        for h5_path in h5_files:
            for density in densities:
                print(f"[RUN] {h5_path.name} @ {density}%")
                result, by_class, frame_counts = run_one_file(
                    source_h5=h5_path,
                    density_pct=density,
                    dataset_name=args.dataset_name,
                    detector_script=args.detector_script,
                    python_exe=args.python,
                    output_dir=args.output_dir,
                    seed=args.seed,
                    temp_root=temp_root,
                    keep_temp_h5=args.keep_temp_h5,
                )

                run_rows.append(
                    {
                        "scene": result.scene,
                        "density_pct": result.density_pct,
                        "input_h5": str(result.input_h5),
                        "detector_input_h5": str(result.detector_input_h5),
                        "output_csv": str(result.output_csv),
                        "n_input_points": result.n_input_points,
                        "n_valid_points": result.n_valid_points,
                        "n_frames": result.n_frames,
                        "n_detections": result.n_detections,
                        "status": result.status,
                        "stderr": result.stderr,
                    }
                )

                for cls, count in by_class.items():
                    class_rows.append(
                        {
                            "scene": result.scene,
                            "density_pct": result.density_pct,
                            "class_label": cls,
                            "class_ID": EXPECTED_CLASS_IDS[cls],
                            "n_detections": count,
                        }
                    )

                if not frame_counts.empty:
                    frame_summaries.append(frame_counts)

                print(f"      -> status={result.status}, detections={result.n_detections}")

        runs_df = pd.DataFrame(run_rows).sort_values(["scene", "density_pct"], ascending=[True, False])
        class_df = pd.DataFrame(class_rows).sort_values(["scene", "density_pct", "class_ID"], ascending=[True, False, True])
        frame_df = pd.concat(frame_summaries, ignore_index=True) if frame_summaries else pd.DataFrame()

        density_df = (
            runs_df.groupby("density_pct", dropna=False)
            .agg(
                n_scenes=("scene", "nunique"),
                n_runs=("scene", "size"),
                n_input_points=("n_input_points", "sum"),
                n_valid_points=("n_valid_points", "sum"),
                n_frames=("n_frames", "sum"),
                n_detections=("n_detections", "sum"),
                n_failed=("status", lambda s: int((s != "ok").sum())),
            )
            .reset_index()
            .sort_values("density_pct", ascending=False)
        )

        class_density_df = (
            class_df.groupby(["density_pct", "class_label", "class_ID"], dropna=False)["n_detections"]
            .sum()
            .reset_index()
            .sort_values(["density_pct", "class_ID"], ascending=[False, True])
        )

        runs_csv = args.output_dir / "scene_density_summary.csv"
        class_csv = args.output_dir / "scene_density_class_summary.csv"
        density_csv = args.output_dir / "density_summary.csv"
        class_density_csv = args.output_dir / "density_class_summary.csv"
        frame_csv = args.output_dir / "frame_detection_summary.csv"
        meta_json = args.output_dir / "run_metadata.json"

        runs_df.to_csv(runs_csv, index=False)
        class_df.to_csv(class_csv, index=False)
        density_df.to_csv(density_csv, index=False)
        class_density_df.to_csv(class_density_csv, index=False)
        if not frame_df.empty:
            frame_df.to_csv(frame_csv, index=False)

        metadata = {
            "input_dir": str(args.input_dir),
            "detector_script": str(args.detector_script),
            "dataset_name": args.dataset_name,
            "densities": densities,
            "seed": args.seed,
            "python": args.python,
            "files": [str(p) for p in h5_files],
            "outputs": {
                "scene_density_summary": str(runs_csv),
                "scene_density_class_summary": str(class_csv),
                "density_summary": str(density_csv),
                "density_class_summary": str(class_density_csv),
                "frame_detection_summary": str(frame_csv) if not frame_df.empty else None,
            },
        }
        meta_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print("\n=== Batch evaluation complete ===")
        print(f"Saved: {runs_csv}")
        print(f"Saved: {class_csv}")
        print(f"Saved: {density_csv}")
        print(f"Saved: {class_density_csv}")
        if not frame_df.empty:
            print(f"Saved: {frame_csv}")
        print(f"Saved: {meta_json}")

        if not density_df.empty:
            print("\nDetections by density:")
            print(density_df.to_string(index=False))

    finally:
        if temp_root_obj is not None:
            temp_root_obj.cleanup()


if __name__ == "__main__":
    main()

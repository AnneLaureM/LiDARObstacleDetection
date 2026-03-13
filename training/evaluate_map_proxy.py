#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from inference.infer_small_point_model import infer_file, load_model
import pandas as pd
import torch

from Hackathon.LiDARObstacleDetection.inference.infer_small_point_model import infer_file, load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, device)

    all_preds = []

    for file_path in sorted(args.input_dir.glob("*.h5")):
        print(f"Processing {file_path.name}")

        pred_df = infer_file(
            model,
            file_path,
            device=device,
            score_threshold=args.score_threshold,
        )

        pred_df["scene"] = file_path.name
        all_preds.append(pred_df)

    if len(all_preds) == 0:
        print("No predictions generated.")
        return

    all_preds = pd.concat(all_preds)

    out_csv = args.output_dir / "all_predictions.csv"
    all_preds.to_csv(out_csv, index=False)

    summary = {
        "num_predictions": len(all_preds),
        "scenes_processed": len(list(args.input_dir.glob("*.h5")))
    }

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved predictions to {out_csv}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
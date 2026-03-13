# README.md

```markdown
# Airbus LiDAR Obstacle Detection
Hybrid Geometry + Lightweight Learning Pipeline

## Overview

This repository implements a **LiDAR obstacle detection system** designed for the Airbus AI Challenge.

The goal is to detect aerial infrastructure obstacles from airborne LiDAR point clouds and output **3D bounding boxes** for the following classes:

- Antenna
- Cable
- Electric Pole
- Wind Turbine

The solution combines:

• **Geometry-based object detection (primary approach)**  
• **Lightweight neural point segmentation (experimental)**  
• **Density robustness evaluation**  
• **Visualization tools for qualitative analysis**

The final submission strategy is:

| Component | Purpose |
|--------|--------|
| V5 heuristic detector | Final prediction CSVs |
| V6 detector | Visual screenshots |
| Small segmentation model | Experimental learning pipeline |
| Proxy evaluation | Internal performance analysis |

The geometry-first approach was chosen because:

- the dataset is small
- objects have strong geometric structure
- cables are difficult for deep models but easy with geometric constraints

---

# Repository Structure

```

airbus_lidar_solution/
├── README.md
├── requirements.txt
├── lidar_utils.py
├── models/
│ └── best_model.pt
├── training/
│ ├── export_dataset_for_segmentation.py
│ ├── train_small_point_model.py
│ ├── evaluate_map_proxy.py
│ └── batch_evaluate_train.py
├── inference/
│ ├── train_and_infer_baseline_v5.py
│ ├── train_and_infer_baseline_v6.py
│ └── infer_small_point_model.py
├── visualization/
│ └── visualize_predictions_v2.py
├── predictions/
│ ├── sceneA_100.csv
│ ├── sceneA_75.csv
│ ├── sceneA_50.csv
│ ├── sceneA_25.csv
│ ├── sceneB_100.csv
│ ├── sceneB_75.csv
│ ├── sceneB_50.csv
│ └── sceneB_25.csv
├── screenshots/
│ ├── frame_01.png
│ ├── ...
│ └── frame_10.png
└── docs/
├── technical_summary.docx
└── one_pager.pdf

````

---

# Installation

Create a Python environment and install dependencies.

```bash
pip install -r requirements.txt
````

---

# Dataset Layout

The training dataset must follow this structure:

```
airbus_hackathon_trainingdata/
├── scene_1.h5
├── scene_2.h5
├── scene_3.h5
├── scene_4.h5
├── scene_5.h5
├── scene_6.h5
├── scene_7.h5
├── scene_8.h5
├── scene_9.h5
└── scene_10.h5
```

Each file contains LiDAR points with:

```
distance_cm
azimuth_raw
elevation_raw
reflectivity
r g b
ego_x ego_y ego_z ego_yaw
```

---

# Final Inference Pipeline (Recommended)

The **V5 heuristic detector** is the recommended final solution.

It is robust to density drops and performs best on **Cable detection**.

Run inference on a single scene:

```bash
python inference/train_and_infer_baseline_v5.py \
--file airbus_hackathon_trainingdata/scene_1.h5 \
--output-csv scene_1_predictions_v5.csv \
--num-workers 8
```

Simulate sparse LiDAR:

```bash
python inference/train_and_infer_baseline_v5.py \
--file airbus_hackathon_trainingdata/scene_1.h5 \
--output-csv scene_1_predictions_v5_50.csv \
--point-fraction 0.5 \
--num-workers 8
```

Diagnostics:

```
--diagnostics-json scene_1_diag.json
```

---

# Batch Robustness Evaluation

Run the detector on the full training set and simulate density variations.

```
python training/batch_evaluate_train.py \
--input-dir airbus_hackathon_trainingdata \
--detector-script inference/train_and_infer_baseline_v5.py \
--output-dir batch_eval_train_v5
```

Inspect summary results:

```
head batch_eval_train_v5/density_summary.csv
head batch_eval_train_v5/density_class_summary.csv
```

These files summarize detection counts across densities:

| density | detections     |
| ------- | -------------- |
| 100%    | baseline       |
| 75%     | robustness     |
| 50%     | sparse LiDAR   |
| 25%     | extreme sparse |

---

# Visualization

Visualize predictions with Open3D.

List available poses:

```bash
python visualization/visualize_predictions_v2.py \
--file airbus_hackathon_trainingdata/scene_1.h5 \
--pred-csv scene_1_predictions_v6.csv
```

Visualize a specific frame:

```bash
python visualization/visualize_predictions_v2.py \
--file airbus_hackathon_trainingdata/scene_1.h5 \
--pred-csv scene_1_predictions_v6.csv \
--pose-index 0
```

Save screenshot:

```bash
python visualization/visualize_predictions_v2.py \
--file airbus_hackathon_trainingdata/scene_1.h5 \
--pred-csv scene_1_predictions_v6.csv \
--pose-index 0 \
--screenshot screenshots/frame_01.png
```

On headless servers:

```bash
xvfb-run -s "-screen 0 1280x720x24" python visualization/visualize_predictions_v2.py ...
```

---

# Screenshot Generation (Airbus Requirement)

Airbus requires **max 10 screenshots** showing:

• point cloud
• predicted bounding boxes
• colored classes

Recommended distribution:

| type         | count |
| ------------ | ----- |
| Cable frames | 3     |
| Wind Turbine | 3     |
| Antenna      | 2     |
| Mixed scene  | 2     |

Total ≤ 10.

Use **V6 predictions** for screenshots since they show more visible objects.

---

# Optional: Segmentation Model Training

Export dataset:

```
python training/export_dataset_for_segmentation.py \
--input-dir airbus_hackathon_trainingdata \
--output-dir seg_dataset \
--max-points 50000
```

Train model:

```
python training/train_small_point_model.py \
--dataset-dir seg_dataset \
--output-dir small_model_runs/run1 \
--epochs 12 \
--batch-size 4
```

Model output:

```
small_model_runs/run1/best_model.pt
```

---

# Optional: Model Evaluation

Proxy evaluation against pseudo labels.

```
python training/evaluate_map_proxy.py \
--model small_model_runs/run1/best_model.pt \
--input-dir airbus_hackathon_trainingdata \
--output-dir proxy_eval_run1
```

This is **not the official Airbus metric**, but useful for development.

---

# Final Submission Generation

When Airbus provides evaluation scenes, run:

```
python inference/train_and_infer_baseline_v5.py \
--file <EVAL_FILE.h5> \
--output-csv <OUTPUT_FILE.csv> \
--num-workers 8
```

Expected final outputs:

```
sceneA_100.csv
sceneA_75.csv
sceneA_50.csv
sceneA_25.csv
sceneB_100.csv
sceneB_75.csv
sceneB_50.csv
sceneB_25.csv
```

---

# Required CSV Format

Each CSV must contain exactly:

```
ego_x
ego_y
ego_z
ego_yaw
bbox_center_x
bbox_center_y
bbox_center_z
bbox_width
bbox_length
bbox_height
bbox_yaw
class_ID
class_label
```

---

# Final Submission Checklist

Before submitting ensure you include:

### Required

* README.md
* requirements.txt
* inference code
* model file
* 8 prediction CSVs
* ≤ 10 screenshots

### Recommended

* technical_summary.docx
* one_page.pdf
* batch evaluation results
* example predictions

---

# Method Summary

The pipeline combines:

1. **Geometric clustering**
2. **PCA line detection**
3. **class-specific heuristics**
4. **optional neural segmentation**

Advantages:

* robust to sparse LiDAR
* interpretable
* efficient
* small model size

This approach performs particularly well on **Cable detection**, the most difficult class in the dataset.

---

# License

Provided for research and competition use.

```

---

# requirements.txt

```

numpy
pandas
scipy
scikit-learn
h5py
torch
tqdm
open3d
matplotlib

```
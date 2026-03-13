# README.md

```markdown
# Airbus LiDAR Obstacle Detection
Hybrid Geometry + Lightweight Learning for Sparse Aerial LiDAR

Author: [Your Name]  
Challenge: Airbus AI Challenge — LiDAR Obstacle Detection

---

# 1. Problem

Airborne LiDAR must detect infrastructure obstacles that can threaten aerial vehicles.

Target objects:

| Class | Description |
|------|-------------|
| Antenna | Vertical communication towers |
| Cable | Thin power or communication lines |
| Electric Pole | Vertical pole infrastructure |
| Wind Turbine | Large wind turbine structures |

The input consists of **sparse LiDAR point clouds** stored in `.h5` files.

The output is **3D bounding boxes** for each detected object.

---

# 2. Solution Overview

We propose a **hybrid pipeline** combining:

- Geometry-based object detection
- PCA-based cable detection
- Density-robust clustering
- Lightweight point segmentation model (optional)

The main insight is that **infrastructure objects have strong geometric signatures**.

This allows reliable detection even under **extreme LiDAR sparsity**.

---

# 3. Pipeline

```

Raw LiDAR (.h5)
│
▼
Spherical → Cartesian conversion
│
▼
Frame extraction (ego poses)
│
▼
Point filtering & downsampling
│
▼
Geometry-based clustering (DBSCAN)
│
▼
Class-specific detection
│
├── Cable detection via PCA line fitting
├── Antenna detection via verticality filters
├── Pole detection via vertical cylinder detection
└── Wind turbine detection via large clusters
│
▼
Bounding box estimation
│
▼
CSV export

```

Optional branch:

```

Pseudo labels
│
▼
Point segmentation model
│
▼
Neural inference
│
▼
Post-processing
│
▼
Bounding boxes

```

---

# 4. Repository Structure

```

LiDARObstacleDetection/

README.md
requirements.txt
lidar_utils.py

training/
export_dataset_for_segmentation.py
train_small_point_model.py
evaluate_map_proxy.py
batch_evaluate_train.py

inference/
train_and_infer_baseline_v5.py
train_and_infer_baseline_v6.py
infer_small_point_model.py

visualization/
visualize_predictions_v2.py

models/
best_model.pt

screenshots/

predictions/

```

---

# 5. Installation

Install dependencies:

```

pip install -r requirements.txt

```

---

# 6. Python Path Configuration

Some scripts import `lidar_utils.py` from the project root.

Run once per terminal session:

```

export PYTHONPATH=$PYTHONPATH:$(pwd)

```

Verify:

```

python -c "import lidar_utils; print('OK')"

```

Expected output:

```

OK

```

---

# 7. Dataset Layout

```

airbus_hackathon_trainingdata/

scene_1.h5
scene_2.h5
...
scene_10.h5

```

Each `.h5` contains LiDAR points with:

```

distance_cm
azimuth_raw
elevation_raw
reflectivity
RGB
ego_x
ego_y
ego_z
ego_yaw

```

---

# 8. Training the Neural Model (Optional)

## Dataset Export

```

python training/export_dataset_for_segmentation.py \
--input-dir airbus_hackathon_trainingdata \
--output-dir seg_dataset \
--max-points 50000 \
--voxel-size 0.15

```

## Train the model

```

python training/train_small_point_model.py \
--dataset-dir seg_dataset \
--output-dir small_model_runs/run1 \ 
--epochs 12 \
--batch-size 4

```

Output:

```

small_model_runs/run1/best_model.pt

```

---

# 9. Batch Robustness Evaluation

Evaluate detector robustness across LiDAR densities.

```

python training/batch_evaluate_train.py \
--input-dir airbus_hackathon_trainingdata \
--detector-script inference/train_and_infer_baseline_v5.py \ 
--output-dir batch_eval_train_v5 \

```

Inspect results:

```

head batch_eval_train_v5/density_summary.csv
head batch_eval_train_v5/density_class_summary.csv

```

Example:

| Density | Detections |
|-------|--------|
|100%|4609|
|75%|4042|
|50%|3244|
|25%|2029|

---

# 10. Inference (Final Detector)

The **V5 heuristic detector** is recommended for final predictions.

```

python inference/train_and_infer_baseline_v5.py \
--file airbus_hackathon_trainingdata/scene_1.h5 \
--output-csv scene_1_predictions_v5.csv \
--num-workers 8

```

Sparse simulation:

```

python inference/train_and_infer_baseline_v5.py \
--file airbus_hackathon_trainingdata/scene_1.h5 \
--output-csv scene_1_predictions_v5_50.csv \
--point-fraction 0.5

```

---

# 11. Visualization

Visualize predictions:

```

python visualization/visualize_predictions_v2.py 
--file airbus_hackathon_trainingdata/scene_1.h5 
--pred-csv scene_1_predictions_v6.csv 
--pose-index 0

```

Generate screenshot:

```

python visualization/visualize_predictions_v2.py 
--file airbus_hackathon_trainingdata/scene_1.h5 
--pred-csv scene_1_predictions_v6.csv 
--pose-index 0 
--screenshot screenshots/frame_01.png

```

On headless servers:

```

xvfb-run -s "-screen 0 1280x720x24" python visualization/visualize_predictions_v2.py ...

```

---

# 12. Screenshots

Airbus requires **≤10 visualization images**.

Recommended distribution:

| Object Type | Images |
|-------------|-------|
Cable | 3
Wind Turbine | 3
Antenna | 2
Mixed scene | 2

---

# 13. Final Submission Generation

For each evaluation scene:

```

python inference/train_and_infer_baseline_v5.py 
--file sceneA_100.h5 
--output-csv sceneA_100.csv

```

Generate:

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

# 14. Output CSV Format

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

# 15. Key Insights

The most difficult class is **Cable**.

Deep neural networks struggle due to:

- extreme sparsity
- thin geometry
- long structures

The geometric approach:

- uses PCA line fitting
- merges aligned clusters
- is robust to LiDAR sparsity

This results in significantly more stable cable detection.

---

# 16. Advantages of the Approach

| Feature | Benefit |
|------|------|
Geometry-first | Robust to sparse LiDAR |
Explainable pipeline | Easy debugging |
Low compute | CPU-friendly |
Hybrid ML | Future improvement possible |

---

# 17. Requirements

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

---

# 18. License

Provided for research and competition use.

---
```

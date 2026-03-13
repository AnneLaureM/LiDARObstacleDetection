# Airbus LiDAR Obstacle Detection

Hybrid Geometry + Lightweight Learning for Sparse Aerial LiDAR

Author: \[Your Name\]\
Challenge: Airbus AI Challenge --- LiDAR Obstacle Detection

------------------------------------------------------------------------

# 1. Problem

Airborne LiDAR must detect infrastructure obstacles that can threaten
aerial vehicles.

Target objects:

  Class           Description
  --------------- -----------------------------------
  Antenna         Vertical communication towers
  Cable           Thin power or communication lines
  Electric Pole   Vertical pole infrastructure
  Wind Turbine    Large wind turbine structures

The input consists of **sparse LiDAR point clouds** stored in `.h5`
files.

The output is **3D bounding boxes** for each detected object.

------------------------------------------------------------------------

# 2. Solution Overview

We propose a **hybrid pipeline** combining:

-   Geometry-based object detection
-   PCA-based cable detection
-   Density-robust clustering
-   Lightweight point segmentation model (optional)

The main insight is that **infrastructure objects have strong geometric
signatures**.

This allows reliable detection even under **extreme LiDAR sparsity**.

------------------------------------------------------------------------

# 3. Pipeline

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

Optional branch:

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

------------------------------------------------------------------------

# 4. Repository Structure

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

------------------------------------------------------------------------

# 5. Installation

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 6. Python Path Configuration

Some scripts import `lidar_utils.py` from the project root.

Run once per terminal session:

``` bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Verify:

``` bash
python -c "import lidar_utils; print('OK')"
```

Expected output:

    OK

------------------------------------------------------------------------

# 7. Dataset Layout

    airbus_hackathon_trainingdata/

    scene_1.h5
    scene_2.h5
    ...
    scene_10.h5

Each `.h5` contains LiDAR points with:

    distance_cm
    azimuth_raw
    elevation_raw
    reflectivity
    RGB
    ego_x
    ego_y
    ego_z
    ego_yaw

------------------------------------------------------------------------

# 8. Training the Neural Model (Optional)

## Dataset Export

``` bash
python training/export_dataset_for_segmentation.py --input-dir airbus_hackathon_trainingdata --output-dir seg_dataset --max-points 50000 --voxel-size 0.15
```

## Train the model

``` bash
python training/train_small_point_model.py --dataset-dir seg_dataset --output-dir small_model_runs/run1 --epochs 12 --batch-size 4
```

Output:

    small_model_runs/run1/best_model.pt

------------------------------------------------------------------------

# 9. Batch Robustness Evaluation

Evaluate detector robustness across LiDAR densities.

``` bash
python training/batch_evaluate_train.py --input-dir airbus_hackathon_trainingdata --detector-script inference/train_and_infer_baseline_v5.py --output-dir batch_eval_train_v5
```

Inspect results:

``` bash
head batch_eval_train_v5/density_summary.csv
head batch_eval_train_v5/density_class_summary.csv
```

------------------------------------------------------------------------

# 10. Inference (Final Detector)

The **V5 heuristic detector** is recommended for final predictions.

``` bash
python inference/train_and_infer_baseline_v5.py --file airbus_hackathon_trainingdata/scene_1.h5 --output-csv scene_1_predictions_v5.csv --num-workers 8
```

Sparse simulation:

``` bash
python inference/train_and_infer_baseline_v5.py --file airbus_hackathon_trainingdata/scene_1.h5 --output-csv scene_1_predictions_v5_50.csv --point-fraction 0.5
```

------------------------------------------------------------------------

# 11. Visualization

``` bash
python visualization/visualize_predictions_v2.py --file airbus_hackathon_trainingdata/scene_1.h5 --pred-csv scene_1_predictions_v6.csv --pose-index 0
```

Generate screenshot:

``` bash
python visualization/visualize_predictions_v2.py --file airbus_hackathon_trainingdata/scene_1.h5 --pred-csv scene_1_predictions_v6.csv --pose-index 0 --screenshot screenshots/frame_01.png
```

Headless servers:

``` bash
xvfb-run -s "-screen 0 1280x720x24" python visualization/visualize_predictions_v2.py ...
```

------------------------------------------------------------------------

# 12. Screenshots

Airbus requires **≤10 visualization images**.

  Object Type    Images
  -------------- --------
  Cable          3
  Wind Turbine   3
  Antenna        2
  Mixed scene    2

------------------------------------------------------------------------

# 13. Output CSV Format

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

------------------------------------------------------------------------

# 14. Requirements

    numpy
    pandas
    scipy
    scikit-learn
    h5py
    torch
    tqdm
    open3d
    matplotlib

------------------------------------------------------------------------

# 15. License

Provided for research and competition use.

# Airbus LiDAR Obstacle Detection

Hybrid Geometry + Lightweight Learning for Sparse Aerial LiDAR

Author: CentraleDigitalLab
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

This work addresses the problem of detecting aerial infrastructure obstacles from airborne LiDAR point clouds in the context of the Airbus AI Challenge. The task consists of identifying and localizing several types of objects that can represent potential hazards for aerial vehicles, namely antennas, cables, electric poles, and wind turbines. The input data is provided as sparse LiDAR scans stored in HDF5 files, where each point is represented in spherical coordinates and associated with reflectivity, RGB color information, and the pose of the sensor platform. The final objective is to produce 3D bounding boxes describing each detected obstacle in a format compatible with the evaluation pipeline of the challenge.

A central observation guiding the design of the solution is that the dataset is relatively limited in size and that the objects of interest exhibit strong geometric regularities. Infrastructure elements such as cables, poles, and antennas are characterized by clear structural properties: cables form long thin linear structures, poles and antennas are predominantly vertical objects with limited cross-sectional area, and wind turbines correspond to large spatial clusters with characteristic dimensions. Because of these properties, purely data-driven deep learning approaches may not be optimal when trained on a limited dataset, especially when the input point clouds are sparse and when some classes, such as cables, correspond to extremely thin geometries that are difficult to capture with standard neural architectures. Training a large deep learning model would also require significantly more labeled data, longer training times, and higher computational resources, while still not guaranteeing robust performance on the most challenging classes.

For these reasons, the core of the proposed approach relies on an algorithmic detection pipeline that exploits the geometric properties of the objects. This strategy allows the system to remain lightweight, interpretable, and computationally efficient while maintaining strong performance under varying point densities. The pipeline begins by converting the raw spherical LiDAR measurements into Cartesian coordinates in the local sensor reference frame. The data is then split into individual frames according to the pose of the LiDAR platform, allowing the detection process to operate on coherent spatial snapshots of the environment. Each frame undergoes filtering and optional downsampling in order to maintain computational efficiency while preserving the geometric structures of interest.

Object hypotheses are generated using spatial clustering techniques applied to the point cloud. Density-based clustering is used to identify groups of points that correspond to potential objects. Once clusters are extracted, class-specific geometric analyses are performed in order to assign object categories and estimate their spatial extent. Linear structures are analyzed using principal component analysis to identify cable-like geometries and to merge aligned clusters into continuous cable segments. Verticality constraints and height-to-width ratios are used to detect poles and antennas. Larger clusters are analyzed to identify wind turbines based on their scale and spatial distribution. For each valid object hypothesis, an oriented 3D bounding box is estimated and exported in the required output format.

An important design consideration in the challenge is robustness to variations in LiDAR density. Airborne sensors may produce point clouds with varying sampling density depending on altitude, motion, and scanning configuration. To ensure that the detector remains reliable in sparse conditions, the pipeline includes mechanisms to simulate reduced densities and evaluate detection stability. The algorithmic nature of the detector proves particularly advantageous in this setting, as geometric reasoning remains effective even when the number of points decreases significantly.

Although the main solution is geometry-based, a complementary machine learning component was also developed in order to explore the potential benefits of neural point classification. A lightweight point-based segmentation network was trained on pseudo-labeled data generated by the heuristic detector. The model contains only a small number of parameters and can be trained quickly, making it suitable for rapid experimentation. While this neural component demonstrates promising point-level classification accuracy, the final detection performance remains primarily driven by the geometric detection stage, especially for thin objects such as cables.

Overall, the proposed solution demonstrates that a carefully designed algorithmic pipeline can outperform heavier learning-based approaches in scenarios where data is limited and object geometries are well structured. By leveraging geometric priors, clustering techniques, and simple statistical analyses, the system achieves reliable detection performance while remaining computationally efficient and easy to train. The final algorithm combines coordinate transformations, density-based clustering, principal component analysis, and class-specific geometric filtering to generate accurate and interpretable 3D obstacle detections.


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

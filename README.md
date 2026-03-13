# Airbus LiDAR Obstacle Detection – Final Submission README

## 1. Overview

This repository implements a **hybrid LiDAR obstacle detection pipeline** for the Airbus challenge.

The target classes are:

- `Antenna`
- `Cable`
- `Electric Pole`
- `Wind Turbine`

The final strategy used here is:

- **v5 heuristic detector for final CSV submission**
- **v6 detector for screenshots / qualitative visualization**
- **small point model for experiments only**
- **proxy evaluation for internal comparison only**

This choice is deliberate:

- `v5` is the most robust on **Cable**, which is the hardest and most important class in this challenge.
- `v6` gives better visual coverage of **Antenna** and **Wind Turbine**, so it is useful for screenshots and demos.
- the lightweight neural model is useful for research and validation, but in the current state it is not the recommended final submission path.

---

## 2. Challenge constraints

Airbus expects a final package including:

1. **8 CSV prediction files**
   - 2 evaluation scenes
   - each at 4 densities:
     - `_100`
     - `_75`
     - `_50`
     - `_25`

2. **A model file**
   - PyTorch (`.pt`, `.pth`) or ONNX

3. **Train and inference code**

4. **requirements.txt**

5. **A README**

6. **Screenshots of max 10 frames**
   - point cloud
   - predicted 3D bounding boxes
   - boxes colored by class

7. **A short technical explanation / slide / one-pager**

---

## 3. Recommended repository structure

```text
airbus_lidar_solution/
├── README.md
├── requirements.txt
├── models/
│   └── best_model.pt
├── training/
│   ├── export_dataset_for_segmentation.py
│   ├── train_small_point_model.py
│   └── evaluate_map_proxy.py
├── inference/
│   ├── train_and_infer_baseline_v5.py
│   ├── train_and_infer_baseline_v6.py
│   └── infer_small_point_model.py
├── visualization/
│   └── visualize_predictions_v2.py
├── screenshots/
│   ├── frame_01.png
│   ├── ...
│   └── frame_10.png
└── examples/
    └── scene_1_predictions.csv
```

---

## 4. Environment setup

Create / activate your environment, then install dependencies:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` is:

```text
numpy
pandas
h5py
scikit-learn
torch
tqdm
open3d
matplotlib
```

---

## 5. Dataset layout

Expected training data layout:

```text
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

Toolkit files expected next to the scripts:

```text
lidar_utils.py
visualize.py
README.md
requirements.txt
```

---

## 6. Recommended final submission path

### Final detector to use for submission

Use:

```text
train_and_infer_baseline_v5.py
```

Reason:

- best Cable behavior
- robust across density drops
- stable on the full train batch evaluation

### Detector to use for screenshots only

Use:

```text
train_and_infer_baseline_v6.py
```

Reason:

- better qualitative visibility for Antenna / Wind Turbine
- more visually convincing screenshots

---

## 7. Run the final heuristic detector on one file

### v5 – recommended final detector

```bash
python train_and_infer_baseline_v5.py   --file airbus_hackathon_trainingdata/scene_1.h5   --output-csv scene_1_predictions_v5.csv   --num-workers 8
```

### v6 – recommended for screenshots / demos

```bash
python train_and_infer_baseline_v6.py   --file airbus_hackathon_trainingdata/scene_1.h5   --output-csv scene_1_predictions_v6.csv   --num-workers 8
```

### Reduced density test

```bash
python train_and_infer_baseline_v5.py   --file airbus_hackathon_trainingdata/scene_1.h5   --output-csv scene_1_predictions_v5_50.csv   --num-workers 8   --point-fraction 0.5   --diagnostics-json scene_1_diag_50.json
```

---

## 8. Full train-batch robustness evaluation

Run the heuristic detector over the whole training set and simulate multiple densities.

### v5 batch evaluation

```bash
python batch_evaluate_train.py   --input-dir airbus_hackathon_trainingdata   --detector-script train_and_infer_baseline_v5.py   --output-dir batch_eval_train_v5
```

### v6 batch evaluation

```bash
python batch_evaluate_train.py   --input-dir airbus_hackathon_trainingdata   --detector-script train_and_infer_baseline_v6.py   --output-dir batch_eval_train_v6
```

### Inspect the summaries

```bash
head -20 batch_eval_train_v5/density_summary.csv
head -20 batch_eval_train_v5/density_class_summary.csv

head -20 batch_eval_train_v6/density_summary.csv
head -20 batch_eval_train_v6/density_class_summary.csv
```

### Interpretation

Use these files to decide which detector is stronger:

- `density_summary.csv`: total detections vs density
- `density_class_summary.csv`: detections per class vs density

Recommended final choice here:

- **submit v5**
- **use v6 only for screenshots**

---

## 9. Export the segmentation dataset (optional experiment path)

This step prepares a point-wise segmentation dataset for the small neural model.

```bash
python export_dataset_for_segmentation.py   --input-dir airbus_hackathon_trainingdata   --output-dir seg_dataset   --max-points 50000   --voxel-size 0.15
```

Expected output:

```text
seg_dataset/
├── train/
├── val/
└── dataset_metadata.json
```

---

## 10. Train the small point model (optional experiment path)

```bash
python train_small_point_model.py   --dataset-dir seg_dataset   --output-dir small_model_runs/run1   --epochs 12   --batch-size 4   --num-workers 4
```

Expected output:

```text
small_model_runs/run1/
├── best_model.pt
├── last_model.pt
└── training_summary.json
```

Check the summary:

```bash
cat small_model_runs/run1/training_summary.json
```

---

## 11. Run model inference (optional experiment path)

```bash
python infer_small_point_model.py   --model small_model_runs/run1/best_model.pt   --file airbus_hackathon_trainingdata/scene_1.h5   --output-csv scene_1_model_predictions.csv
```

---

## 12. Proxy evaluation of the model (optional experiment path)

This is **internal validation only**. It is not the Airbus official evaluation, but it is useful for development.

```bash
python evaluate_map_proxy.py   --model small_model_runs/run1/best_model.pt   --input-dir airbus_hackathon_trainingdata   --output-dir proxy_eval_run1
```

Inspect outputs:

```bash
cat proxy_eval_run1/summary.json
head -30 proxy_eval_run1/all_predictions.csv
```

Recommendation:

- keep this script for internal experiments
- do **not** rely on it as the official challenge metric
- do **not** use the current model alone as the final submission path

---

## 13. Generate screenshots for Airbus

Airbus asks for **max 10 screenshots** showing:

- the point cloud
- predicted 3D bounding boxes
- bounding boxes colored by class

### List available poses in a scene

```bash
python visualize_predictions_v2.py   --file airbus_hackathon_trainingdata/scene_1.h5   --pred-csv scene_1_predictions_v6.csv
```

### Visualize one pose

```bash
python visualize_predictions_v2.py   --file airbus_hackathon_trainingdata/scene_1.h5   --pred-csv scene_1_predictions_v6.csv   --pose-index 0
```

### Save one screenshot

```bash
python visualize_predictions_v2.py   --file airbus_hackathon_trainingdata/scene_1.h5   --pred-csv scene_1_predictions_v6.csv   --pose-index 0   --screenshot screenshots/scene1_pose0.png
```

### On a headless server

```bash
xvfb-run -s "-screen 0 1280x720x24" python visualize_predictions_v2.py   --file airbus_hackathon_trainingdata/scene_1.h5   --pred-csv scene_1_predictions_v6.csv   --pose-index 0   --screenshot screenshots/scene1_pose0.png
```

### Suggested screenshot selection

Produce **10 screenshots max** total:

- 3 frames with clear **Cable**
- 3 frames with clear **Wind Turbine**
- 2 frames with **Antenna**
- 2 mixed frames

Recommended:
- use **v6 outputs** for screenshots
- use multiple densities if possible (`100`, `50`, `25`) to show robustness

---

## 14. Generate final evaluation CSVs for Airbus

Once Airbus provides the 8 evaluation files, run the final detector on each file.

### Recommended final inference command

```bash
python train_and_infer_baseline_v5.py   --file <EVAL_FILE.h5>   --output-csv <OUTPUT_FILE.csv>   --num-workers 8
```

### Expected final files

You must produce something like:

```text
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

## 15. Required CSV format

Each prediction CSV must contain these columns exactly:

```text
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

## 16. Final submission checklist

Before zipping the repository, check that you have:

### Required
- [ ] `README.md`
- [ ] `requirements.txt`
- [ ] final inference script
- [ ] training code
- [ ] model file (`.pt`, `.pth`, or `.onnx`)
- [ ] 8 CSV prediction files
- [ ] max 10 screenshots
- [ ] short technical summary / slide / PDF

### Recommended
- [ ] one-pager with method explanation
- [ ] parameter count of the model
- [ ] runtime notes
- [ ] screenshots generated with `visualize_predictions_v2.py`

---

## 17. Recommended final package content

A good final submission zip could contain:

```text
submission/
├── README.md
├── requirements.txt
├── inference/
│   └── train_and_infer_baseline_v5.py
├── training/
│   ├── export_dataset_for_segmentation.py
│   ├── train_small_point_model.py
│   └── evaluate_map_proxy.py
├── visualization/
│   └── visualize_predictions_v2.py
├── models/
│   └── best_model.pt
├── predictions/
│   ├── sceneA_100.csv
│   ├── sceneA_75.csv
│   ├── sceneA_50.csv
│   ├── sceneA_25.csv
│   ├── sceneB_100.csv
│   ├── sceneB_75.csv
│   ├── sceneB_50.csv
│   └── sceneB_25.csv
├── screenshots/
│   ├── frame_01.png
│   ├── ...
│   └── frame_10.png
└── docs/
    ├── technical_summary.docx
    └── one_pager.pdf
```

---

## 18. Final recommendation

### Use for final submission:
- `train_and_infer_baseline_v5.py`

### Use for screenshots / qualitative visualization:
- `train_and_infer_baseline_v6.py`
- `visualize_predictions_v2.py`

### Keep for internal experiments:
- `export_dataset_for_segmentation.py`
- `train_small_point_model.py`
- `infer_small_point_model.py`
- `evaluate_map_proxy.py`

This hybrid workflow gave the best balance between:

- Cable robustness
- density robustness
- low parameter count
- engineering stability

For this challenge, **the final practical choice is to submit the v5 heuristic detector** and keep the model path as supporting work.
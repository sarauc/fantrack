# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FANTrack is a 3D multi-object tracking system using a Feature Association Network. It tracks objects (cars, pedestrians) in the KITTI dataset by combining a Similarity Network (SimNet) and an Association Network (AssocNet).

## Environment

- Python 3.6.5, TensorFlow 1.13.1 (GPU), CUDA 10.0, CuDNN 7.6.4
- Install dependencies: `pip install -r requirements.txt`
- Requires pre-trained AVOD object detector weights extracted to `fan_track/object_detector/avod/data/`

## Common Commands

```bash
# Train SimNet (similarity network)
python3 fan_track/experiments/train_simnet.py

# Train AssocNet (association network)
python3 fan_track/experiments/train_assocnet.py

# Run tracker inference
python3 fan_track/experiments/run_tracker.py
```

## Architecture

### Two-stage neural network pipeline

1. **SimNet** (`network/model.py`) — Similarity network with two branches:
   - Bounding box branch: processes 3D box geometry
   - Appearance branch: processes CNN features from AVOD
   - Both branches project features onto a unit hypersphere; cosine similarity drives matching

2. **AssocNet** (`network/model.py`) — Association network:
   - Takes a 160×160 global correlation map as input (plus 21×21 local crops per target)
   - Uses dilated convolutions (rates 2, 4, 6) to predict pixel-level target-measurement associations
   - Also estimates detection probability (probability of existence)

3. **Tracker** (`network/tracker.py`, ~1000 lines) — Top-level inference loop:
   - Loads AVOD 3D detections per frame
   - Runs Kalman filter prediction
   - Calls SimNet + AssocNet for data association
   - Maintains track state across frames

### Data flow

KITTI raw data → AVOD 3D detector → feature extraction (`avod_feature_extractor/`) → correlation maps (`data_generation/map_generator.py`) → SimNet + AssocNet inference → tracked objects

### Key configuration (`fan_track/config/config.py`)

- `GlobalConfig`: KITTI paths, map dimensions (160×160), max targets/measurements (21)
- `SimnetConfig`: Training hyperparameters for SimNet (500 epochs, Adam lr=1e-4)
- `AssocnetConfig`: Training hyperparameters for AssocNet (exponential LR decay, momentum 0.9)
- `TrackerConfig`: Kalman filter params, AVOD score thresholds, visualization flags

### Training data generation

- `data_generation/kitti_simnet_dataset.py` — generates positive/negative bounding box pairs
- `data_generation/kitti_assocnet_dataset.py` — creates association labels from KITTI ground truth
- `data_generation/simnet_batch_sampler.py` / `batch_sampler.py` — balanced mini-batch sampling

### Checkpoints

- SimNet checkpoints → `fan_track/data/simnet/`
- AssocNet checkpoints → `fan_track/data/assocnet/`
- Pre-trained AVOD VGG weights → `fan_track/avod_feature_extractor/checkpoints/`

## Known Compatibility Notes

The notebook `FanTrack_Instructions_Github.ipynb` documents 9 source modifications needed for Colab/newer environments (e.g., float precision fixes in `object_detector/calib_utils.py`, AVOD path configuration, batch normalization toggle). Check this notebook before debugging import or path errors.

## FANTrack: 3D Multi-Object Tracking with Feature Association Network

### Requirements
Please refer to this [requirement.txt](requirement.txt)

### Environment Setup
Please refer to this [notebook](FanTrack.ipynb)

### Training

1. **AVOD**
FANTrack requires trained weights for AVOD. You can find our pre-trained weights [avod_cars_fast.zip](wiselab.uwaterloo.ca/avod/avod_cars_fast.zip) (138 Mb) and [avod_people_fast.zip](wiselab.uwaterloo.ca/avod/avod_people_fast.zip) (163 Mb). Unzip this folder into `fan_track/object_detector/avod`. It should produce a folder  `fan_track/object_detector/avod/data`. Alternatively, you can train AVOD yourself using the [AVOD instructions](https://github.com/kujason/avod).

2. **SimNet**
  Train simnet by running:
  ```bash
  $ python3 fan_track/experiments/train_simnet.py
  ```
  Checkpoints and data will be saved to `fan_track/data/simnet`

3. **AssocNet**
Train assocnet by running:
  ```bash
  $ python3 fan_track/experiments/train_simnet.py
  ```
    
4. **Run tracker**
  ```bash
  $ python3 fan_track/experiments/run_tracker.py
  ```

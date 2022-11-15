## FANTrack: 3D Multi-Object Tracking with Feature Association Network

### Requirements
1. GPU that supports CUDA.
2. tensorflow-gpu 1.13
3. CUDA 10.0
4. cuDNN v7.6.5 for CUDA 10.0
5. scipy
6. numpy
7. opencv-python
8. matplotlib
9. scikit-learn
10. filterpy
11. tqdm

### Setup
---
Please refer to this [notebook](FanTrack.ipynb)

### Training
---
1. **AVOD**

    FANTrack requires trained weights for AVOD. You can find our pre-trained weights [avod_cars_fast.zip](wiselab.uwaterloo.ca/avod/avod_cars_fast.zip) (138 Mb) and [avod_people_fast.zip](wiselab.uwaterloo.ca/avod/avod_people_fast.zip) (163 Mb). Unzip this folder into `fan_track/object_detector/avod`.

    It should produce a folder  `fan_track/object_detector/avod/data`

    Alternatively, you can train AVOD yourself using the [AVOD instructions](https://github.com/kujason/avod).

3. **SimNet**

    Train simnet by running:
    ```bash
    $ python3 fan_track/experiments/train_simnet.py
    ```

    Checkpoints and data will be saved to `fan_track/data/simnet`

4. **AssocNet**

    Train assocnet by running:
    ```bash
    $ python3 fan_track/experiments/train_simnet.py
    ```

    Checkpoints and data will be saved to `fan_track/data/assocnet`

You can modify SimNet and AssocNet hyperparameters in `fan_track/config/config.py`


### Inference
---

Run Inference using:
```bash
$ python3 fan_track/experiments/run_tracker.py
```

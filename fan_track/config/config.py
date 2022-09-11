import os
from pathlib import Path

def fan_track_dir():
    """Returns project root folder."""
    return str(Path(__file__).parent.parent.resolve())


class GlobalConfig:
    """
        GlobalConfig has configuratins that are generic to both the networks.
    """

    KITTI_ROOT = '/content/Kitti'  #os.path.expanduser('~/Kitti')
    KITTI_DIR = KITTI_ROOT + '/object'
    SIMNET_DATASET_PATH = KITTI_ROOT + '/simnet_dataset'

    # BoundingBox Appearance Minibatches Path
    SIMNET_MINIBATCH_PATH = KITTI_ROOT + '/simnet_minibatches'

    ASSOCNET_DATASET = KITTI_ROOT + '/assocnet_dataset'
    TRACKING_DATASET = KITTI_ROOT + '/tracking_dataset'
    OBJECT_DETECTION_DATASET = '~/obj_det'

    # Global map length
    MAP_LENGTH = 160

    # Global map width
    MAP_WIDTH = 160
    MAX_TARGETS = 21
    MAX_MEASUREMENTS = 21

    # Local map crop
    CROP_SIZE = 21

    OPTIMIZER = 'Adam'

class SimnetConfig(GlobalConfig):
    """
        Simnet specific configurations
    """

    SIMNET_CKPT_DIR = os.path.join(fan_track_dir(), 'data/simnet/checkpoints')
    SIMNET_CKPT_PATH = os.path.join(SIMNET_CKPT_DIR, 'simnet.ckpt-499')
    EPOCHS = 500
    VALIDATION_RATE = 1
    SAVE_RATE = 10
    ACCURACY_THRESHOLD = 0.0
    LEARNING_RATE = 1e-4

    # Weight decay constant used for regularization
    WEIGHT_DECAY = 1e-3

    MAX_CHECKPOINTS = 0

class AssocnetConfig(GlobalConfig):
    """
        Assocnet specific configurations
    """

    ASSOCNET_CKPT_DIR = os.path.join(fan_track_dir(), 'data/assocnet/checkpoints')
    ASSOCNET_CKPT_PATH = os.path.join(ASSOCNET_CKPT_DIR, 'assocnet.ckpt-999')
    LEARNING_RATE = 1e-4

    # Decay rate for exponentially decaying learning rate
    DECAY_RATE = 0.95
    DECAY_STEP = 10

    # Momentum term used in momentum optimization
    MOMENTUM = 0.9

    # Weight decay constant used for regularization
    WEIGHT_DECAY = 1.0

class TrackerConfig(GlobalConfig):
    """
       Tracker specific configurations
    """

    # Start Frame - Defaults to None for tracking all the frames
    START_FRAME = None
    END_FRAME = None

    # cars Checkpoint 00221000 threshold 0.28
    # Pedestrian checkpoint 00120000 threshold 0.52
    # AVOD_CKPT_NAME is 'pyramid_cars_with_aug_example' for Cars and 'pyramid_people_example' for people

    EXISTENCE_THRESHOLD = 0.40
    SURVIVABILITY_FACTOR = 0.60
    DETECTOR_THRESHOLD = 0.28
    AVOD_CKPT_NAME = 'pyramid_cars_with_aug_example'
    AVOD_CKPT_NUMBER = '00221000'

    # length of the observation history [frames]
    OBS_HIS_LENGTH= 50

    # AVOD Uncertainties for Kalman Filter
    AVOD_FAR=0.026008
    AVOD_X_MSE=0.051589
    AVOD_Y_MSE=0.013226
    AVOD_Z_MSE=0.010612
    AVOD_H_MSE=0.125582
    AVOD_W_MSE=0.009738
    AVOD_L_MSE=0.008613

    DT = 0.1

    # Accelaration factor for Q matrix of Kalman Filter
    ACCELERATION_VARIANCE_FACTOR = 1

    # Visualize images
    # Set VISUALIZE_RANGE for visualization to work for specific range of frames
    VISUALIZE = True
    VISUALIZE_PREDICTIONS = False
    VISUALIZE_RANGE = range(0,1000)

    # Use GroundTruth Detections (Only for Training)
    USE_GT = False

    # GPU Device
    DEVICE_ID = '0'

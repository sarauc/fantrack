import gc
import os
import time
import sys
import numpy as np
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import fan_track.utils.generic_utils as gu
import tensorflow as tf
from fan_track.network.tracker import Tracker
from fan_track.utils.generic_utils import trim_txt_files
from fan_track.config.config import GlobalConfig

def main(_):
    start_time = time.time()
    args = gu.prepare_args()

    # For Cars
    args.avod_checkpoint_name = 'pyramid_cars_with_aug_example'
    # For People
    # args.avod_checkpoint_name = 'pyramid_people_example'

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    trim_txt_files(GlobalConfig.KITTI_ROOT) # Trim whitespace from the end of lines. Otherwise, messes up the CSV parser

    for i in range(0,21):
        video_no = "%04d" % i
        tracker = Tracker(args, video_no, dataset_name='training')
        tracker.begin_tracking(video_no, dataset_subtype='val')
        del tracker

    elapsed_time = time.time() - start_time
    print('Time taken:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

if __name__ == '__main__':
    tf.app.run()

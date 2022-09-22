import os
import numpy as np
import sys
import time
import shutil
import fan_track.data_generation.position_shape_utils as ps_utils
import fan_track.utils.kitti_tracking_utils as kitti_tracking_utils
import tqdm
from fan_track.data_generation.kitti_imu_to_cam import cam_to_imu_transform
from fan_track.data_generation.kitti_imu_to_cam import oxts_prev_frame_trans
from distutils.dir_util import copy_tree
from fan_track.avod_feature_extractor.bev_cam_feature_extractor import BevCamFeatExtractor
from fan_track.config.config import GlobalConfig
from functools import reduce
from enum import Enum
from avod import root_dir as avod_root_dir
from avod.builders.dataset_builder import DatasetBuilder
import avod.builders.config_builder_util as config_builder

# occlusion states
class occlusion(Enum):
    FULLY_VISIBLE = '0'
    PARTLY_OCCLUDED = '1'
    LARGELY_OCCLUDED = '2'
    UNKNOWN = '3'

# occluded states
occluded = (occlusion.LARGELY_OCCLUDED.value,occlusion.UNKNOWN.value)

# global space
object_types = ('Car', 'Van', 'Pedestrian',  'Cyclist')

# the dimension of the Kitti's 3d bbox: [dimx,dimy,dimz,x,y,z,ry]
bbox_len = 7

class KittiSimnetDataset():

    def __init__(self):

        # the ratio of validation to training
        self.validation_ratio = 0.20

        avod_root_dir = '/content/fantrack/fan_track/object_detector/'
        checkpoint_name = 'avod_cars_fast'
        experiment_config = checkpoint_name + '.config'
        experiment_config_path = os.path.join(avod_root_dir ,'data',checkpoint_name,experiment_config)


        # Read the configurations
        model_config, _, _, dataset_config = config_builder.get_configs_from_pipeline_file(
            experiment_config_path, is_training=False)



        # Overwrite the defaults
        self.dataset_config = config_builder.proto_to_obj(dataset_config)
        self.dataset_config.data_split = 'test'
        self.dataset_config.data_split_dir = 'testing'
        self.dataset_config.has_labels = False
        self.dataset_config.dataset_dir = GlobalConfig.KITTI_ROOT + "/object"

        # Remove augmentation during evaluation in test mode
        self.dataset_config.aug_list = []

        kitti_dataset = DatasetBuilder.build_kitti_dataset(self.dataset_config, use_defaults=False)

        self.feat_ext = BevCamFeatExtractor(dataset=kitti_dataset, model_config=model_config)
        self.feat_ext.prepare_inputs()
        self.feat_ext.setup_bev_img_vggs()
        self.feat_ext.setup_expand_predictions()

        self.gen_transf_matrices()

        self.file_count = {}

    def generate_dataset(self, label_files = None):
        '''Generate numpy array files for objects in each two successive frames in a video '''
       
        #/content/Kitti/tracking_dataset/labels/label_02 --> about 20 samples 
        label_path = os.path.join(GlobalConfig.TRACKING_DATASET,
                                                                 'labels',
                                                                 'label_02')
        if (label_files is None):
            # obtain the label files
            label_files = get_filenames(label_path)

        # total number of positive and negative examples
        total_pos_exp_consecutive = 0
        total_neg_exp_consecutive = 0
        total_pos_exp_oddeven = 0
        total_neg_exp_oddeven = 0

        main_progressbar = tqdm.tqdm(total=len(label_files), desc='Video', position=0)
        # create numpy files holding examples and save them
        for i,file in enumerate(label_files):

            # the name of the video file
            video_no = os.path.splitext(file)[0]

            self.dataset_dir_training = create_dataset_dir(video_no, is_training_frame=True)
            self.dataset_dir_validation = create_dataset_dir(video_no, is_training_frame=False)

            # Determine the number of frames in the video
            kitti_img_path = os.path.join(GlobalConfig.KITTI_ROOT,
                                          'tracking_dataset', 'image_2' ,'training',video_no)
            _, _, frames = next(os.walk(kitti_img_path))
            self.num_frames = len(frames)

            # prepare the inputs to the avod
            prepare_data(video_no)    # for each new video rebuild the dataset

            # Build the kitti dataset object
            self.kitti_dataset = DatasetBuilder.build_kitti_dataset(self.dataset_config, use_defaults=False)
            self.feat_ext.dataset = self.kitti_dataset

            num_pos_exp, num_neg_exp = self.create_examples_odd_even(label_path, file,video_no)

            total_pos_exp_consecutive += num_pos_exp
            total_neg_exp_consecutive += num_neg_exp

            num_pos_exp, num_neg_exp = self.create_examples(label_path,file,video_no)

            total_pos_exp_oddeven += num_pos_exp
            total_neg_exp_oddeven += num_neg_exp

            main_progressbar.update(1)
        main_progressbar.close()

        print('\n')
        print('Consecutive: Total number of positive examples is {0:d}\n'
              'Consecutive: Total number of negative examples is {1:d}'
              .format(total_pos_exp_consecutive, total_neg_exp_consecutive))

        print('Consecutive: Total number of positive examples is {0:d}\n'
              'Consecutive: Total number of negative examples is {1:d}'
              .format(total_pos_exp_oddeven,total_neg_exp_oddeven))

    def create_examples_odd_even(self, label_path, filename, video_no):
        '''
            Create the positive and negative examples for each object found in
            a given video file.
            Inputs: label_path is the path to the label directory
                    filename is the txt file containing labels.
                    video_no is the video number in the format of
                    'xxxx'.
                    feat_ext is the avod feature extractor.
        '''

        is_training_frame = True
        last_training_frame = int(self.num_frames * (1 - self.validation_ratio))

        # target measurement pairs from odd frames
        targets_odd = np.empty((0, 7), dtype=np.float32)
        t_labels_odd = []
        t_types_odd = []
        t_occluded_odd = []

        measurements_odd = np.empty((0, 7), dtype=np.float32)
        m_labels_odd = []
        m_types_odd = []
        m_occluded_odd = []

        # target measurement pairs from even frames
        targets_even = np.empty((0, 7), dtype=np.float32)
        t_labels_even = []
        t_types_even = []
        t_occluded_even = []

        measurements_even = np.empty((0, 7), dtype=np.float32)
        m_labels_even = []
        m_types_even = []
        m_occluded_even = []

        total_pos_exp_video = 0
        total_neg_exp_video = 0

        frame_id = None

        # the index of the example
        if video_no not in self.file_count:
            example_idx = 0
        else:
            example_idx = self.file_count[video_no]

        # flag for even frame pairs
        even_frame = True

        try:
            with open(label_path + '/' + filename, 'r') as f:

                # read each line of label file
                for line_no, line in enumerate(f):

                    # split the string on whitespace to obtain a list of columns
                    obj_info = line.split()

                    if (line_no == 0 and frame_id is None):
                        frame_id = int(obj_info[0])
                    else:
                        if (frame_id < int(obj_info[0])):

                            # check if at least two frames are already processed
                            if (frame_id > 1 and frame_id != last_training_frame+1):

                                # determine which pairs to use
                                if (even_frame):
                                    targets = targets_even
                                    t_labels = t_labels_even
                                    t_types = t_types_even
                                    t_occluded = t_occluded_even

                                    measurements = measurements_even
                                    m_labels = m_labels_even
                                    m_types = m_types_even
                                    m_occluded = m_occluded_even
                                else:
                                    targets = targets_odd
                                    t_labels = t_labels_odd
                                    t_types = t_types_odd
                                    t_occluded = t_occluded_odd

                                    measurements = measurements_odd
                                    m_labels = m_labels_odd
                                    m_types = m_types_odd
                                    m_occluded = m_occluded_odd

                                if (measurements.size > 0):

                                    if (targets.size > 0):
                                        
                                        # kitti dataset issue
                                        if video_no == '0001' and frame_id in range(177, 181):
                                          print ('There is missing data in KITTI tracking dataset at seq 1, frame 177-180!')
                                          continue
                                            
                                        # save targets, measurements, and labels in a numpy file
                                        example_idx, num_pos_exp, num_neg_pos,m_keep = self.save_examples(video_no,
                                                                                 example_idx,
                                                                                 targets,
                                                                                 measurements,
                                                                                 t_labels,
                                                                                 m_labels,
                                                                                 t_types,
                                                                                 m_types,
                                                                                 t_occluded,
                                                                                 m_occluded,
                                                                                 frame_id,
                                                                                 is_training_frame,
                                                                                 is_odd_even=False)

                                        total_pos_exp_video += num_pos_exp
                                        total_neg_exp_video += num_neg_pos

                                    # keep measurements for the next target-measurement pair
                                    else:
                                        m_keep = True
                                else:
                                    m_keep = False

                                if (m_keep):
                                    # save current measurements as targets and clear measurements for the next frame
                                    if (frame_id % 2 == 0):
                                        targets_even = measurements
                                        t_labels_even = m_labels[:]
                                        t_types_even = m_types
                                        t_occluded_even = m_occluded

                                    else:
                                        targets_odd = measurements
                                        t_labels_odd = m_labels[:]
                                        t_types_odd = m_types
                                        t_occluded_odd = m_occluded

                                # create a new measurement matrix and clear the contents of label and type lists
                                if (even_frame):
                                    measurements_even = np.empty((0, 7), dtype=np.float32)
                                    m_labels_even[:] = []
                                    m_types_even = []
                                    m_occluded_even = []
                                else:
                                    measurements_odd = np.empty((0, 7), dtype=np.float32)
                                    m_labels_odd[:] = []
                                    m_types_odd = []
                                    m_occluded_odd = []

                                if frame_id == last_training_frame:
                                    targets_odd = np.empty((0, 7), dtype=np.float32)
                                    targets_even = np.empty((0, 7), dtype=np.float32)
                                    t_labels_odd = []
                                    t_labels_even = []
                                    t_types_odd = []
                                    t_occluded_odd = []
                                    t_types_even = []
                                    t_occluded_even = []

                                    measurements_odd = np.empty((0, 7), dtype=np.float32)
                                    measurements_even = np.empty((0, 7), dtype=np.float32)
                                    m_labels_odd = []
                                    m_labels_even = []
                                    m_types_odd = []
                                    m_types_even = []
                                    m_occluded_odd = []
                                    m_occluded_even = []

                                    even_frame = True

                            frame_id = int(obj_info[0])

                    if (obj_info[2] in object_types):

                        # get the bbox params in the format [h,w,l,x,y,z,ry]
                        bbox_ry = np.asarray([obj_info[10:17]], dtype=np.float32)

                        # convert Kitti's box to 3d box format [x,y,z,l,w,h,ry]
                        bbox_ry = ps_utils.kitti_box_to_box_3d(bbox_ry)

                        # unique track id of the object within this sequence
                        track_id = obj_info[1]

                        # type of the object
                        obj_type = obj_info[2]

                        occluded = obj_info[4]

                        # add bbox params and rotation_y parameters of the target
                        if (frame_id == 0 or frame_id == last_training_frame+1):
                            targets_even = np.append(targets_even, bbox_ry, axis=0)
                            t_labels_even.append(track_id)
                            t_types_even.append(obj_type)
                            t_occluded_even.append(occluded)

                        elif (frame_id == 1 or frame_id == last_training_frame+2):
                            targets_odd = np.append(targets_odd, bbox_ry, axis=0)
                            t_labels_odd.append(track_id)
                            t_types_odd.append(obj_type)
                            t_occluded_odd.append(occluded)

                        elif (frame_id % 2 == 0 or frame_id - (last_training_frame+1) % 2 == 0):
                            measurements_even = np.append(measurements_even, bbox_ry, axis=0)
                            m_labels_even.append(track_id)
                            m_types_even.append(obj_type)
                            m_occluded_even.append(occluded)

                            even_frame = True
                        else:
                            measurements_odd = np.append(measurements_odd, bbox_ry, axis=0)
                            m_labels_odd.append(track_id)
                            m_types_odd.append(obj_type)
                            m_occluded_odd.append(occluded)

                            even_frame = False

            #print('video no:{0:}, number of examples:{1:d}'.format(video_no, example_idx))

        except IOError as e:
            print('Could not open the file {0.filename}'.format(e))
            sys.exit()

        return total_pos_exp_video, total_neg_exp_video

    def create_examples(self, label_path,filename,video_no):
        '''
            Create the positive and negative examples for each object found in
            a given video file.
            Inputs: label_path is the path to the label directory
                    filename is the txt file containing labels.
                    video_no is the video number in the format of
                    'xxxx'.
                    feat_ext is the avod feature extractor.
        '''

        is_training_frame = True
        last_training_frame = int(self.num_frames * (1 - self.validation_ratio))

        # targets and measurement matrices
        targets =  np.empty((0,7),dtype=np.float32)
        t_labels = []
        t_types = []
        t_occluded = []

        measurements = np.empty((0,7),dtype=np.float32)
        m_labels = []
        m_types = []
        m_occluded = []
        frame_id = None

        # the index of the example
        example_idx = 0

        # total number of positive and negative examples for a given video
        total_pos_exp_video = 0
        total_neg_exp_video = 0
        num_lines = 0

        # Read the number of lines for progressbar
        with open(label_path + '/' + filename,'r') as f:
            for item in f:
                num_lines += 1

        with open(label_path + '/' + filename,'r') as f:
            bar = tqdm.tqdm(total=num_lines, position=3, desc="Frame:")
            # read each line of label file
            for line_no, line in enumerate(f):

                # split the string on whitespace to obtain a list of columns
                obj_info = line.split()

                if (frame_id is None):
                    frame_id = int(obj_info[0])
                else:
                    if (frame_id < int(obj_info[0])):

                        # check if at least two frames are already processed (in training or validation)
                        if (frame_id != 0 and frame_id != last_training_frame +1):

                            if frame_id > last_training_frame:
                                is_training_frame = False
                                
                            # kitti dataset issue
                            if video_no == '0001' and frame_id in range(177, 181):
                                print ('There is missing data in KITTI tracking dataset at seq 1, frame 177-180!')
                                continue
                                            
                            # save targets, measurements, and labels in the numpy file named as example_idx
                            example_idx, num_pos_exp,num_neg_pos,_ = self.save_examples(video_no,
                                                                                       example_idx,
                                                                                       targets,
                                                                                       measurements,
                                                                                       t_labels,
                                                                                       m_labels,
                                                                                       t_types,
                                                                                       m_types,
                                                                                       t_occluded,
                                                                                       m_occluded,
                                                                                       frame_id,
                                                                                       is_training_frame,
                                                                                       is_odd_even = True
                                                                                      )

                            self.file_count[video_no] = example_idx

                            # add number of positive and negative samples to the totals for the video
                            total_pos_exp_video += num_pos_exp
                            total_neg_exp_video += num_neg_pos

                            # save current measurements as targets and clear measurements for the next frame
                            targets = measurements
                            t_labels[:] = m_labels[:]
                            t_types[:] = m_types[:]
                            t_occluded[:] = m_occluded[:]

                            if frame_id == last_training_frame:
                                targets = np.empty((0, 7), dtype=np.float32)
                                t_labels = []
                                t_types = []
                                t_occluded = []

                            # create a new measurement matrix and clear the contents of label and type lists
                            measurements = np.empty((0,7),dtype=np.float32)
                            m_labels[:] = []
                            m_types[:] = []
                            m_occluded = []

                        frame_id = int(obj_info[0])

                if (obj_info[2] in object_types):

                    # get the bbox params in the format [h,w,l,x,y,z,ry]
                    bbox_ry = np.asarray([obj_info[10:17]], dtype=np.float32)

                    # convert Kitti's box to 3d box format [x,y,z,l,w,h,ry]
                    bbox_ry = ps_utils.kitti_box_to_box_3d(bbox_ry)

                    # unique track id of the object within this sequence
                    track_id = obj_info[1]

                    # type of the object
                    obj_type = obj_info[2]

                    occluded = obj_info[4]

                    # Check if first frame in training or first frame in validation
                    if (frame_id == 0 or frame_id == last_training_frame+1):
                        # add bbox params and rotation_y parameters of the target
                        targets = np.append(targets, bbox_ry, axis=0)
                        t_labels.append(track_id)
                        t_types.append(obj_type)
                        t_occluded.append(occluded)
                    else:
                        measurements = np.append(measurements, bbox_ry, axis=0)
                        m_labels.append(track_id)
                        m_types.append(obj_type)
                        m_occluded.append(occluded)
                bar.update(1)
        bar.write('video no:{0:}, number of examples:{1:d}'.format(video_no,example_idx))

        bar.write('total number of positive examples in this video is {0:d}\n'
              'total number of negative examples in this video is {1:d}'.format(total_pos_exp_video,
                                                                                total_neg_exp_video))

        return total_pos_exp_video, total_neg_exp_video

        # except IOError as e:
        #     print('Could not open the file {0.filename}'.format(e))
        #     sys.exit()

    def save_examples(self, video_no, example_idx, targets, measurements, t_labels,
                      m_labels,t_types, m_types, t_occluded, m_occluded, meas_frame, is_training_frame, is_odd_even):
        '''
            Save the object at frame k as target, and the one at frame k+1 as measurement
            together with the label indicating if they are associated or not, measurement
            frame no, and their fused features from the detector avod in a numpy file.
            Return: example_idx is the index used to name files for each video.
                    num_pos_pair is the number of positive example pairs.
                    num_neg_pair is the number of negative example pairs.
        '''

        # save numpy files under this directory for each video
        if is_training_frame:
            save_path = self.dataset_dir_training
        else:
            save_path = self.dataset_dir_validation

        # number of positive and negative pairs to generate
        if not is_training_frame or is_odd_even:
            num_pairs = 1
        else:
            num_pairs = 5

        num_pos_pair = 0
        num_neg_pair = 0

        target_frame = meas_frame-1

        # retrieve inputs to extract targets' features
        self.feat_ext.get_sample(target_frame)

        # prepare anchors for the avof
        self.feat_ext.get_anchors(targets)

        # extract bev and cam features
        t_features = self.feat_ext.extract_features()

        # obtain target features
        # t_features = feat_ext.early_fusion()

        # retrieve inputs to extract the measurements' features
        self.feat_ext.get_sample(meas_frame)

        Z_stack = np.empty((0,7),dtype=np.float32)

        num_neg_match = [0]*targets.shape[0]
        num_pos_match = [0]*targets.shape[0]

        for idx_t,t in enumerate(targets):

            # find the corresponding measurement for the given target t
            m_t, m_n = self.find_pos_index(m_labels,t_labels[idx_t])

            pos_matches = np.empty((0,7),dtype=np.float32)
            neg_matches = np.empty((0,7),dtype=np.float32)

            if not(m_t is None):

                #print('Generating positive pairs')

                # corresponding measurement to target t
                z = measurements[m_t,:]

                # determine object type for iou threshold
                type_idx = get_obj_type_idx(m_types[m_t])

                # generate positive matches by augmentation
                pos_matches = self.augmentation(z, type_idx, num_pairs)

                Z_stack = np.concatenate((Z_stack,pos_matches), axis=0)

                num_pos_match[idx_t] = num_pairs

                num_pos_pair += num_pairs

            # indices of other detections with the same type
            if ( not(m_n is None) and not(t_occluded[idx_t] in occluded) ):
                m_n = [j for j in m_n if  m_types[j] == t_types[idx_t]  ]
            else:
                # no other detections are available
                m_n = []

            # if there are other detections with the same type
            if (len(m_n)>0):

                #print('Generating negative pairs')

                # negative pair counter
                neg_cntr = 0

                neg_matches = np.zeros(shape=(num_pairs,7), dtype=np.float32)

                # generate negative pairs
                while(neg_cntr<num_pairs):

                    # randomly pick another measurement to generate negative example
                    m = np.random.choice(m_n,1)[0]

                    # use it as target to generate a negative match
                    z = measurements[m,:]

                    # determine object type for iou threshold
                    type_idx = get_obj_type_idx(m_types[m])

                    # generate a negative match
                    neg_matches[neg_cntr] = self.augmentation(z, type_idx, 1)

                    neg_cntr += 1

                num_neg_pair += num_pairs

                num_neg_match[idx_t] = num_pairs

            Z_stack = np.concatenate((Z_stack,neg_matches), axis=0)

        if (Z_stack.shape[0] > 0):

            # prepare anchors for the avod
            self.feat_ext.get_anchors(Z_stack)

            # extract bev and cam features
            m_features = self.feat_ext.extract_features()

            targets

            # obtain avod's fused features
            # m_features = feat_ext.early_fusion()

            for idx_t,t in enumerate(targets):

                t_feat_idx_t = t_features[idx_t]

                num_pos = num_pos_match[idx_t]
                num_neg = num_neg_match[idx_t]

                # no pair for this target
                if (num_pos == 0 and num_neg == 0):
                    continue

                # transform the target's 3d bbox from cam coordinates to imu_gps coordinates
                self.transform_cam_to_imu(video_no, meas_frame-1, t, is_target = True)

                for idx_m,z in enumerate(Z_stack):

                    if (idx_m - num_pos == num_neg):
                        break
                    # transform the measurement's 3d bbox params from cam coordinates to imu_gps coordinates
                    meas_imu_k = self.transform_cam_to_imu(video_no, meas_frame, z, is_target = False)

                    if (idx_m < num_pos):
                        simnet_label = 1

                    elif (idx_m - num_pos < num_neg):
                        simnet_label = -1

                    # Savetonpy
                    # assign a unique index to each example by increasing the example_idx
                    example_idx += 1

                    # generate example
                    example = {'target': t,
                               'measurement': z,
                               'target_feat': t_feat_idx_t,
                               'meas_feat': m_features[idx_m],
                               'simnet_label': simnet_label,
                               'video_no': video_no,
                               'meas_imu_k': meas_imu_k,
                               'meas_frame': meas_frame
                               }

                    file_name = str(example_idx).rjust(7, '0') + '.npy'
                    np.save(os.path.join(save_path, file_name), example)

                Z_stack = Z_stack[idx_m:]
                m_features = m_features[idx_m:]

        return example_idx, num_pos_pair, num_neg_pair, True

    def augmentation(self, bbox_ry, type_idx, num_samples = 5):
        '''
            Augment the 3d bbox coordinates of the given example to generate
            new positive examples.
            Inputs: The 3d bbox and rotation y (r_y) vector, and  the type
            of the object.
            Output: The matrix of the 3d_bbox and r_y of the augmented
            positive examples.
        '''

        # iou criteria to consider augmented detections as objects
        iou_pos_Th = (0.75,0.80) # cars and pedestrian/cyclist respectively

        # the discrimination threshold between positive examples
        iou_dis_Th = 0.95

        # the matrix of positive examples
        pos_examples = np.zeros(shape=(num_samples,7),dtype=np.float32)

        # augmentation of the 3d bbox along dimx, dimy and dimz with laplace distribution
        loc, sc  = 1, 0.05

        # augmentation of the 3d bbox by rotating around the rotation ry
        upper_bound = np.pi/36
        lower_bound = -1 * upper_bound

        # augmentation of the 3d bbox at the center points x,y,z
        mu = bbox_ry[0:3]

        dimxyz = ps_utils.box_3d_dimxyz(bbox_ry)

        # assuming that 0.5*dim is equal to the 3*sigma
        sigma = dimxyz/6

        for i in range(num_samples):

            # create a new copy of bbox parameters
            pos_examples[i] = np.copy(bbox_ry)

            # data augmentation flag
            aug_accept =  False

            cntr = 0

            while not(aug_accept):

                cntr += 1

                if (cntr > 1):
                    # reinitialize positive example
                    pos_examples[i][:] = bbox_ry[:]

                # the first example is the original true example
                if (i == 0):
                    aug_ind = np.zeros(3)
                else:
                    # randomly select the augmentations:
                    aug_ind = np.random.choice([0,1],3)

                    # neglect no augmentation
                    while not(np.any(aug_ind)):
                        aug_ind = np.random.choice([0,1],3)

                # apply the augmentations
                for idx,j in enumerate(aug_ind):

                    if (j == 1):
                        # scale augmentation
                        if(idx == 0):
                            sxyz = np.random.laplace(loc, sc, size=3)
                            pos_examples[i][3:-1] = np.multiply(bbox_ry[3:-1],sxyz)

                        # rotation augmentation
                        elif(idx == 1):
                            theta = np.random.uniform(lower_bound,upper_bound,size=1)
                            pos_examples[i][-1] = bbox_ry[-1] + theta

                        # center augmentation
                        else:
                            pos_examples[i][0:3] = np.random.normal(mu, sigma, 3)

                # object must be in front of the camera
                if (pos_examples[i][2] < 0):
                    pos_examples[i][2] = mu[2]

                # check the IoU threshold is satisfied
                iou = ps_utils.bev_iou(bbox_ry,pos_examples[i], self.kitti_dataset)

                if (iou >= iou_pos_Th[type_idx]):

                    aug_accept = True

                    # check the discrimination threshold for positive examples
                    for prev_exp in pos_examples[0:i]:

                        iou = ps_utils.bev_iou(prev_exp,pos_examples[i],self.kitti_dataset)

                        # neglect the example as it is too similar to an existing one
                        if (iou > iou_dis_Th):
                            aug_accept = False
                            break;

        return pos_examples

    def find_pos_index(self, m_labels,t_label):
        '''
            Find the same label as the target label and other labels in the measurement
            labels.
            Return: a tuple of the indices consisting of the target label index
            and other indices if found in measurement labels. Otherwise, if any
            of them found, return None for those.
        '''

        M = len(m_labels)

        if (t_label in m_labels):

            # index of the target label
            m_t = m_labels.index(t_label)

            # indices of other labels
            m_n = [i for i in range(M) if i is not m_t]

            if ( len(m_n) > 0 ):
                return m_t, m_n
            else:
                return m_t, None

        else:
            # indices of negative examples
            m_n = [i for i in range(M)]

            if ( len(m_n) > 0 ):
                return None, m_n
            else:
                return None, None

    def transform_cam_to_imu(self, video_no, frame_number, bbox, is_target):
        '''
            Transform 3d bounding box positions of measurement and target
            in the kth and (k-1)st camera coordinates, respectively into the
            (k-1)st imu_gps coordinates. In addition, compute rotation_ry of
            measurement with respect to the orientation of the camera at time
            k-1.
            Input: bbox is the 3d bbox params in the format [x,y,z,l,h,ry].
                   is_target is a boolean used to determine transformation
                   is for a target or measurement.
            Output: 3d bbox params of the measurement in the kth imu_gps co-
                   ordinates.
        '''

        x_cam_k = bbox.copy()

        # transform x,y,z at the kth camera coordinates into the kth imu_gps coordinates
        x_imu_k = np.dot(self.trans_matrices[video_no],np.append(bbox[:3], 1))

        meas_imu_k = None

        if not(is_target): # transformation into (k-1)st imu_gps coordinates

            # copy the measurement's centers in the kth imu_gps coordinates for mapping
            meas_imu_k = x_imu_k[0:3].copy()

            # transform x,y,z in the kth imu_gps coordinates to the (k-1)st imu_gps coordinates
            x_imu_prev = np.dot(self.oxts_projections[video_no][frame_number],x_imu_k)

            # check if x center is positive, i.e., object is ahead of the imu/gps
            if (x_imu_prev[0] < 0):
                raise('objects cannot be behind the car')

            bbox[:3] = x_imu_prev[:-1]

            # transform rotation ry at time k wrt the orientation of the cam at time k-1
            bbox[-1] -= self.delta_yaws[video_no][frame_number]
            dim_xyz = self.box_dimen_transform(video_no, x_cam_k, frame_number, False)

        else: # transformation into the kth imu_gps coordinates

            bbox[:3] = x_imu_k[:-1]
            dim_xyz = self.box_dimen_transform(video_no, x_cam_k, frame_number, True)

        # dimensions in (k-1)st the imu_gps coordinates
        bbox[3:-1] = dim_xyz[:]

        return meas_imu_k

    def box_dimen_transform(self, video_no, bbox_3d, k, t_flag):
        '''
            Transform the dimensions of the bbox in the kth camera coordinates
            into the (k-1)st imu_gps coordinates.
            Input: bbox_3d is the 3d box in the format of [x,y,z,l,w,h,ry]
                   k is the frame no.
                   t_flag is the boolean target flag.
            Output: 1d numpy array including new dimensions in the (k-1)st
                    imu_gps coordinates. The elements of the array is in
                    the format of [dx,dy,dz]. Note that x pointing forward,
                    y pointing left, and z pointing up in the imu_gps coor-
                    dinates.
        '''

        # bbox center at x,y,z coordinates
        x = bbox_3d[0]
        y = bbox_3d[1]
        z = bbox_3d[2]

        # bbox length, width, and height
        l = bbox_3d[3]
        w = bbox_3d[4]
        h = bbox_3d[5]

        # x,y,z components of the 8 corners
        x_corners = [x + l/2, x + l/2,
                     x - l/2, x - l/2,
                     x + l/2, x + l/2,
                     x - l/2, x - l/2]

        y_corners = [y,   y,
                     y,   y,
                     y-h, y-h,
                     y-h, y-h ]

        z_corners = [z + w/2, z - w/2,
                     z - w/2, z + w/2,
                     z + w/2, z - w/2,
                     z - w/2, z + w/2 ]

        # create a ones column
        ones_col = np.ones((1,len(x_corners)))

        # corner points in the kth camera coordinates
        corners = np.asarray([x_corners,
                              y_corners,
                              z_corners], dtype = np.float64)

        # transform corners into the kth imu_gps coordinates
        c_imu = np.dot(self.trans_matrices[video_no], np.append(corners,ones_col,axis=0))

        if not(t_flag):
            # transform the corners into the (k-1)st imu_gps coordinates
            c_imu = np.dot( self.oxts_projections[video_no][k],c_imu)

        # new dimensions in the (k-1)st imu_gps coordinates
        dim_xyz = 2*np.mean(np.absolute(c_imu[:-1,:] - np.mean(c_imu[:-1,:],axis=1, keepdims=True)),axis=1)

        # check if any bbox dimension is negative
        assert reduce(lambda x,y: x*y,dim_xyz) > 0, 'bbox dimension cannot be negative'

        return dim_xyz

    def gen_transf_matrices(self):
        '''
            For each video generate transformation matrices from camera
            coordinates to imu_gps coordinates and imu_gps transformation
            matrices from frame k to frame k-1.
        '''

        self.trans_matrices = {}
        self.oxts_projections = {}
        self.delta_yaws = {}

        # calibration directory of training videos
        calib_dir = os.path.join(GlobalConfig.TRACKING_DATASET,
                                                 'calib',
                                                 'training')

        # get calibration files for each videos
        _, _,calib_files = next(os.walk(calib_dir))

        for f in calib_files:

            video_no = os.path.splitext(f)[0]

            # path to the calibration file
            path_calib = os.path.join(calib_dir, f)

            # transformation matrix from camera to the first imu coordinates
            self.trans_matrices[video_no] = cam_to_imu_transform(path_calib)

            # transformation matrices from ith oxts coord to 1st oxts coord
            self.oxts_projections[video_no], self.delta_yaws[video_no] = oxts_prev_frame_trans(video_no=int(video_no))

def prepare_data(video_no):
    '''
        Copy calib, plane, image_2, and velodyne files.
        In addition, write frame numbers in test.txt
    '''

    # kitti input path
    kitti_input_path = ps_utils.kitti_dir +  \
                      '/{0}/{1}'.format('object', 'testing')

    # subdirectories under the kitti's input directory
    dirpath, folders, _ = next(os.walk(kitti_input_path))

    # first copy velodyne point cloud files
    i = folders.index('velodyne')
    folders[0],folders[i] = folders[i],folders[0]

    # reset frames each new video video
    kitti_tracking_utils.frames = None

    for f in folders:

        folder_path = os.path.join(dirpath, f)

        files = os.listdir(folder_path)

        # delete files under the folder
        [os.remove(folder_path + '/' + f) for f in files]

        if (f == 'calib'):

            kitti_calib_path = GlobalConfig.KITTI_ROOT +  \
                               '/{0}/{1}/{2}'.format('tracking_dataset','calib','training')

            src_calib_path = os.path.join(kitti_calib_path,video_no + '.txt')

            shutil.copy(src_calib_path, folder_path)

            kitti_tracking_utils.duplicate_calib()

        elif (f == 'planes'):

            kitti_tracking_utils.create_planes()

        elif(f == 'image_2'):

            kitti_img_2_path = GlobalConfig.KITTI_ROOT +  \
                               '/{0}/{1}/{2}'.format('tracking_dataset','image_2','training')

            src_img_2_path = os.path.join(kitti_img_2_path,video_no)

            # copy images in src directory to the folder_path
            copy_tree(src_img_2_path, folder_path)

        elif(f == 'velodyne'):

            kitti_velodyne_path = GlobalConfig.KITTI_ROOT +  \
                                  '/{0}/{1}/{2}'.format('tracking_dataset','velodyne','training')

            src_velodyne_path = os.path.join(kitti_velodyne_path,video_no)

            # copy images in src directory to the folder_path
            copy_tree(src_velodyne_path, folder_path)

    # clean the test.txt and rewrite the frame numbers
    kitti_tracking_utils.rewrite_data_split_file()

def get_filenames(path_to_files):
    '''
       Find all files in a directory tree.
       Input: folders where the files reside under the
       Kitti directory, .i.e., '~/Kitti/folder1/folder2'.
       Return: The filenames.
    '''

    print(path_to_files)
    _, _, files = next(os.walk(path_to_files))
    files = sorted(files)

    return files

def create_dataset_dir(folder_name, is_training_frame):
    '''
        Create dataset directory to save numpy files.
        Return: directory where to save examples.
    '''

    if is_training_frame:
        training_dir = os.path.join(GlobalConfig.SIMNET_DATASET_PATH, folder_name, 'training')

        if not os.path.exists(training_dir):
            os.makedirs(training_dir)

        return training_dir

    else:
        validation_dir = os.path.join(GlobalConfig.SIMNET_DATASET_PATH, folder_name, 'validation')

        if not os.path.exists(validation_dir):
            os.makedirs(validation_dir)

        return validation_dir

def get_obj_type_idx(obj_type):

    if (obj_type == object_types[2] or
        obj_type == object_types[3]):
        return 1

    else:
        return 0

if __name__ == '__main__':

    # Set the path for the dataset in config in GlobalConfig.BB_APP_DS_PATH

    start_time = time.time()
    dataset_obj = KittiSimnetDataset()
    dataset_obj.generate_dataset()
    dataset_obj.feat_ext.close_sessions()

    elapsed_time = time.time() - start_time
    print('Time taken:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

import os
import numpy as np
import shutil
import sys
import fan_track.data_generation.position_shape_utils as ps_utils
from fan_track.utils import kitti_tracking_utils
from fan_track.data_generation.kitti_imu_to_cam import cam_to_imu_transform
from fan_track.data_generation.kitti_imu_to_cam import imu_to_cam_transform
from fan_track.data_generation.kitti_imu_to_cam import oxts_prev_frame_trans
from fan_track.data_generation.kitti_imu_to_cam import oxts_0_frame_trans
from fan_track.config.config import GlobalConfig
from distutils.dir_util import copy_tree
from fan_track.avod_feature_extractor.bev_cam_feature_extractor import BevCamFeatExtractor
from functools import reduce
from avod import root_dir as avod_root_dir
from fan_track.data_generation.map_generator import MapGenerator
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
import tqdm

if sys.version_info > (3, 0):
    python3=True
else:
    python3=False

# global space
object_types = ('Car', 'Van', 'Pedestrian',  'Cyclist')

# the dimension of the Kitti's 3d bbox: [dimx,dimy,dimz,x,y,z,ry]
bbox_len = 7

global feat_ext

class KittiAssocnetDataset(MapGenerator):

    def __init__(self, cardinality_T=21, cardinality_M=21, crop_size=21, dataset_name = 'training'):

        super(KittiAssocnetDataset, self).__init__()

        self.dataset_name = dataset_name

        self.gen_transf_matrices()

        # maximum number of targets and measurements in one frame
        self.card_T = cardinality_T

        self.card_M = cardinality_M

        # region to crop around every target
        self.crop_size = crop_size

        # Ratio of validation to training
        self.validation_split_ratio = 0.20

        checkpoint_name = 'pyramid_cars_with_aug_example'
        experiment_config = checkpoint_name + '.config'
        experiment_config_path = os.path.join(avod_root_dir(), 'data',
                                              'outputs', checkpoint_name, experiment_config)

        # Read the configurations
        model_config, _, _, dataset_config = config_builder.get_configs_from_pipeline_file(
            experiment_config_path, is_training=False)

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

    def generate_dataset(self, label_files = None):
        '''Generate numpy array files for objects in each two successive frames in a video '''

        label_path = os.path.join(GlobalConfig.TRACKING_DATASET,
                                                                 'labels',
                                                                 'label_02')
        if (label_files is None):
            # obtain the label files
            label_files = get_filenames(label_path)

        # create the main folder including all examples
        create_dataset_dir('')

        global feat_ext

        status_log = tqdm.tqdm(total=0, position=2, bar_format='{desc}')
        main_progressbar = tqdm.tqdm(total=len(label_files), desc='Video', position=1)
        # create numpy files holding examples and save them
        for i,file in enumerate(label_files):

            # the name of the video file
            video_no = os.path.splitext(file)[0]
            status_log.set_description_str('Video: {}, Preparing AVOD input...'.format(video_no))
            # prepare the inputs to the avod
            prepare_data(video_no)    # for each new video rebuild the dataset

            kitti_img_2_path = os.path.join(GlobalConfig.TRACKING_DATASET,'image_2',self.dataset_name,video_no)

            if python3:
                _, _, frames = next(os.walk(kitti_img_2_path))
            else:
                _, _, frames = os.walk(kitti_img_2_path).next()

            self.num_frames = len(frames)
            status_log.set_description_str('Video: {}, Building KITTI Dataset...'.format(video_no))

            self.kitti_dataset = DatasetBuilder.build_kitti_dataset(self.dataset_config, use_defaults=False)
            self.feat_ext.dataset = self.kitti_dataset

            status_log.set_description_str('Video: {}, Creating AssocNet Dataset...'.format(video_no))
            self.create_examples(label_path,file,video_no)

            main_progressbar.update(1)
        main_progressbar.close()
        status_log.close()

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

        # targets and measurement matrices
        targets =  np.empty((0,7),dtype=np.float32)
        t_labels = []

        measurements = np.empty((0,7),dtype=np.float32)
        m_labels = []

        frame_id = None

        # the index of the example
        example_idx = 0
        num_lines = 0
        try :
            # Read the number of lines for progressbar
            with open(label_path + '/' + filename,'r') as f:
                for item in f:
                    num_lines += 1

            with open(label_path + '/' + filename,'r') as f:
                bar = tqdm.tqdm(total=num_lines, position=0, desc='Frame')
                # read each line of label file
                for line_no, line in enumerate(f):
                    bar.update(1)

                    # split the string on whitespace to obtain a list of columns
                    obj_info = line.split()

                    if (line_no == 0 and frame_id is None):
                        frame_id = int(obj_info[0])
                    else:
                        if (frame_id < int(obj_info[0])):

                            # check if at least two frames are already processed
                            if (frame_id > 0):

                                if (measurements.size > 0):

                                    if (targets.size > 0):

                                        if frame_id > int(self.num_frames * (1- self.validation_split_ratio)):
                                            train_val = 'validation'
                                        else:
                                            train_val = 'training'
                                        try:
                                            # save targets, measurements, and labels in a numpy file
                                            example_idx, m_keep = self.save_examples(video_no,
                                                                                    example_idx,
                                                                                    targets,
                                                                                    measurements,
                                                                                    t_labels,
                                                                                    m_labels,
                                                                                    frame_id,
                                                                                    train_val
                                                                                    )
                                        except ValueError as e:
                                            bar.write("ERROR parsing training data for video {} example {}".format(video_no, example_idx))
                                            m_keep = False
                                            example_idx += 1
                                    # keep measurements for the next target-measurement pair
                                    else:
                                        m_keep = True

                                else:
                                    m_keep = False

                                if (m_keep):

                                    # save current measurements as targets and clear measurements for the next frame
                                    targets = measurements
                                    t_labels[:] = m_labels[:]


                                # create a new measurement matrix and clear the contents of label and type lists
                                measurements = np.empty((0,7),dtype=np.float32)
                                m_labels[:] = []



                            frame_id = int(obj_info[0])


                    if (obj_info[2] in object_types):

                        # get the bbox params in the format [h,w,l,x,y,z,ry]
                        bbox_ry = np.asarray([obj_info[10:17]], dtype=np.float32)

                        # convert Kitti's box to 3d box format [x,y,z,l,w,h,ry]
                        bbox_ry = ps_utils.kitti_box_to_box_3d(bbox_ry)

                        # unique track id of the object within this sequence
                        track_id = obj_info[1]


                        if (frame_id == 0):
                            # add bbox params and rotation_y parameters of the target
                            targets = np.append(targets, bbox_ry, axis=0)
                            t_labels.append(track_id)

                        else:
                            measurements = np.append(measurements, bbox_ry, axis=0)
                            m_labels.append(track_id)


            #print('video no:{0:}, number of examples:{1:d}'.format(video_no,example_idx))

        except IOError as e:
            print('Could not open the file {0.filename}'.format(e))
            sys.exit()

    def save_examples(self, video_no, example_idx, targets, measurements, t_labels, m_labels, meas_frame, train_val):
        '''
            Save the object at frame k as target, and the one at frame k+1 as measurement
            together with the label indicating if they are associated or not, measurement
            frame no, and their fused features from the detector avod in a numpy file.
            Return: example_idx is the index used to name files for each video.
                    num_pos_pair is the number of positive example pairs.
                    num_neg_pair is the number of negative example pairs.
        '''

        # save numpy files under this directory for each video

        training_video_dir, validation_video_dir = create_dataset_dir(video_no)

        # copy 3d bbox coordinates of targets to extract features
        c_targets = np.copy(targets)

        # copy 3d bbox coordinates of measurements to extract
        # features and use them as targets in the next frame
        c_measurements = np.copy(measurements)

        target_frame = meas_frame-1

        # measurements at imu coordinates at time k
        Z_imu_k = np.empty((0,3),dtype=np.float32)

        for t in c_targets:

                # transform the targe30t's 3d bbox from cam coordinates to imu_gps coordinates
                self.transform_cam_to_imu(video_no, meas_frame-1, t, is_target = True)

        # target centers
        t_xyz = np.asarray([t[0:3] for t in c_targets],np.float32)

        # map targets and compute coordinates of local maps
        self.map_targets(t_xyz)

        # check if there exists any mapped-targets
        targets_exist = reduce(lambda x,y: x or y, self.mapped_objects)

        # return example_idx and the flag for keeping measurements as targets
        if not(targets_exist):
            return example_idx, True

        # remove targets which are out of the map
        targets_feat = targets[self.mapped_objects]

        targets_bbox = c_targets[self.mapped_objects]

        # labels of mapped targets
        t_labels = [x[0] for x in zip(t_labels,self.mapped_objects) if x[1]]

        for z in c_measurements:

                # transform 3d bbox from cam coordinates to imu-gps coordinates at target time
                self.transform_cam_to_imu(video_no, meas_frame, z, is_target = False)

                Z_imu_k  = np.concatenate((Z_imu_k, z[0:3].reshape(1,-1)), axis=0)

        # locate measurements and compute coordinates of local maps
        z_xyz = np.asarray([z[0:3] for z in Z_imu_k],np.float32)

        self.loc_ind_coords(z_xyz)

        # check if there exists any mapped-measurements
        measurements_exist = reduce(lambda x,y: x or y, self.mapped_objects)

        # return example_idx and the flag for keeping measurements as targets
        if not(measurements_exist):
            return example_idx, False

        # labels of mapped measurements
        m_labels = [x[0] for x in zip(m_labels,self.mapped_objects) if x[1]]

        # remove measurements which are out of the map
        measurements_feat = measurements[self.mapped_objects]

        measurements_bbox = c_measurements[self.mapped_objects]

        # create label maps for the second assocNet
        label_maps = np.zeros((1,self.crop_size,self.crop_size,self.card_T))

        # targets are centered at local maps
        tar_pos_idx = int((self.crop_size - 1)/2 + 1)

        for idx_t, label in enumerate(t_labels):
            # find the corresponding measurement for the given target t
            i, _ = self.find_pos_index(m_labels,label)

            if not(i is None):
                # positional indices of measurement locations according to the left-bottom corner
                vert_pos_idx = tar_pos_idx - int((self.t_centers[idx_t][0] - self.m_centers[i][0]))
                horz_pos_idx = tar_pos_idx - int((self.t_centers[idx_t][1] - self.m_centers[i][1]))

                label_maps[0][vert_pos_idx, horz_pos_idx, idx_t] = 1

        # expand dimension for simnet to [None,1,2]
        self.t_centers = np.expand_dims(self.t_centers, axis=1)

        # feature extractor
        global feat_ext

        # retrieve inputs to extract targets' features
        self.feat_ext.get_sample(target_frame)

        # prepare anchors for the avof
        self.feat_ext.get_anchors(targets_feat)

        # extract bev and cam features
        t_features = self.feat_ext.extract_features()

        # obtain target features
        #t_features = self.feat_ext.early_fusion()

        # retrieve inputs to extract to measurements' features
        self.feat_ext.get_sample(meas_frame)

        # prepare anchors for the avod
        self.feat_ext.get_anchors(measurements_feat)

        # extract bev and cam features
        m_features = self.feat_ext.extract_features()

        # obtain avod's fused features
        #m_features = self.feat_ext.early_fusion()

        # assign a unique index to each example by increasing the example_idx
        example_idx += 1

        # generate example
        example = {'targets': targets_bbox,
                   'measurements': measurements_bbox,
                   'target_feat': t_features,
                   'meas_feat': m_features,
                   'targets_xy': self.t_centers,
                   'measurements_xy': self.m_centers,
                   'labels': label_maps,
                   'num_targets': targets_bbox.shape[0],
                   'video_no': video_no,
                   'meas_locations': self.meas_loc,
                   'meas_frame': meas_frame
                   }

        file_name = str(example_idx).rjust(7,'0') + '.npy'

        if train_val == 'training':
            file_path =  training_video_dir + '/{0}'.format(file_name)
        else:
            file_path = validation_video_dir + '/{0}'.format(file_name)

        np.save(file_path, example)

        # return example_idx and False for keeping targets
        return example_idx, True

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

    def transform_cam_to_imu(self, video_no, frame_number, bbox, is_target, toimu0=False):
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

        # transform x,y,z at the kth camera coordinates into the kth imu_gps coordinates
        xyz1_imu_k = np.dot(self.trans_camtoimu_matrices[video_no], np.append(bbox[:3], 1))

        # If we need to convert to 0th imu_gps coordinates
        if toimu0 and frame_number != 0:
            xyz1_imu_k = np.dot(self.oxts_to0_projections[video_no][frame_number],xyz1_imu_k)
            dim_xyz = self.box_dimen_transform(video_no, bbox, frame_number, t_flag=False, to_imu0=True)
            yaw_0 = self.delta_yaws[video_no][1]
            bbox[-1] -= yaw_0

        elif not(is_target): # transformation into (k-1)st imu_gps coordinates

            # copy the measurement's centers in the kth imu_gps coordinates for mapping
            # meas_imu_k = x_imu_k[0:3].copy()

            # transform x,y,z in the kth imu_gps coordinates to the (k-1)st imu_gps coordinates
            xyz1_imu_k = np.dot(self.oxts_projections[video_no][frame_number],xyz1_imu_k)

            # check if x center is positive, i.e., object is ahead of the imu/gps
            if (xyz1_imu_k[0] < 0):
                raise('objects cannot be behind the car')

            dim_xyz = self.box_dimen_transform(video_no, bbox, frame_number, False)

            # transform rotation ry at time k wrt the orientation of the cam at time k-1
            bbox[-1] -= self.delta_yaws[video_no][frame_number]

        else: # transformation into the kth imu_gps coordinates
            dim_xyz = self.box_dimen_transform(video_no, bbox, frame_number, True)

        # bbox centers in the imu-gps coordinates
        bbox[:3] = xyz1_imu_k[:-1]

        # dimensions in the imu_gps coordinates
        bbox[3:-1] = dim_xyz[:]

    def transform_imu0_to_imuk(self,video_no, frame_number, bbox):
        # transform x,y,z in the 0th imu_gps coordinates to the kth imu_gps coordinates
        imu_k = np.dot(self.oxts_0tok_projections[video_no][frame_number], np.append(bbox[:3], 1))

        return imu_k

    def transform_imu0_to_cam(self, video_no, frame_number, bbox):

        # transform x,y,z in the 0th imu_gps coordinates to the kth imu_gps coordinates
        imu_k = self.transform_imu0_to_imuk(video_no, frame_number, bbox)

        # transform x,y,z at the kth imu_gps coordinates into kth camera coordinates
        cam_k = np.dot(self.trans_imutocam_matrices[video_no], imu_k)

        # bbox centers in the 3D camera coordinates
        bbox[:3] = cam_k[:-1]

        yaw_0 = self.delta_yaws[video_no][1]
        bbox[-1] += yaw_0

        # Transform the dimensions from kth imu gps to kth camera coordinates
        dim_xyz = self.box_dimen_transform_imu0_to_cam(video_no, bbox, frame_number)

        # dimensions in the imu_gps coordinates
        bbox[3:-1] = dim_xyz[:]

    def box_dimen_transform(self, video_no, bbox_3d, k, t_flag, to_imu0 = False):
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
        x_corners = [x + l / 2, x + l / 2,
                     x - l / 2, x - l / 2,
                     x + l / 2, x + l / 2,
                     x - l / 2, x - l / 2]

        y_corners = [y, y,
                     y, y,
                     y - h, y - h,
                     y - h, y - h]

        z_corners = [z + w / 2, z - w / 2,
                     z - w / 2, z + w / 2,
                     z + w / 2, z - w / 2,
                     z - w / 2, z + w / 2]

        # create a ones column
        ones_col = np.ones((1, len(x_corners)))

        # corner points in the kth camera coordinates
        corners = np.asarray([x_corners,
                              y_corners,
                              z_corners], dtype=np.float64)

        # transform corners into the kth imu_gps coordinates
        c_imu = np.dot(self.trans_camtoimu_matrices[video_no], np.append(corners, ones_col, axis=0))

        if to_imu0:
            c_imu = np.dot(self.oxts_to0_projections[video_no][k], c_imu)
        elif not (t_flag):
            # transform the corners into the (k-1)st imu_gps coordinates
            c_imu = np.dot(self.oxts_projections[video_no][k], c_imu)

        # new dimensions in the (k-1)st imu_gps coordinates
        dim_xyz = 2 * np.mean(np.absolute(c_imu[:-1, :] - np.mean(c_imu[:-1, :], axis=1, keepdims=True)), axis=1)

        # check if any bbox dimension is negative
        assert reduce(lambda x, y: x * y, dim_xyz) > 0, 'bbox dimension cannot be negative'

        return dim_xyz

    def box_dimen_transform_imu0_to_cam(self, video_no, bbox_3d, k):
        '''
          imu0 dimensions to camk dimensions
        '''

        # bbox center at x,y,z coordinates
        x = bbox_3d[0]
        y = bbox_3d[1]
        z = bbox_3d[2]

        # bbox length, width, and height
        l = bbox_3d[4]
        w = bbox_3d[3]
        h = bbox_3d[5]

        # x,y,z components of the 8 corners
        x_corners = [x + w / 2, x + w / 2,
                     x - w / 2, x - w / 2,
                     x + w / 2, x + w / 2,
                     x - w / 2, x - w / 2]

        y_corners = [y + l / 2, y - l / 2,
                     y - l / 2, y + l / 2,
                     y + l / 2, y - l / 2,
                     y - l / 2, y + l / 2]

        z_corners = [z, z,
                     z, z,
                     z - h, z - h,
                     z - h, z - h]

        # create a ones column
        ones_col = np.ones((1, len(x_corners)))

        # corner points in the 0th imu_gps coordinates
        corners = np.asarray([x_corners,
                              y_corners,
                              z_corners], dtype=np.float64)

        # transform corners into kth imu_gps coordinates
        c_imu = np.dot(self.oxts_0tok_projections[video_no][k], np.append(corners, ones_col, axis=0))

        # transform corners into the kth camera coordinates
        c_cam = np.dot(self.trans_imutocam_matrices[video_no],c_imu)

        # new dimensions in the (k-1)st imu_gps coordinates
        dim_xyz = 2 * np.mean(np.absolute(c_cam[:-1, :] - np.mean(c_cam[:-1, :], axis=1, keepdims=True)), axis=1)

        # swap h and w to be in cam coordinate format l w h
        dim_xyz[1], dim_xyz[2] = dim_xyz[2], dim_xyz[1]

        # check if any bbox dimension is negative
        assert reduce(lambda x, y: x * y, dim_xyz) > 0, 'bbox dimension cannot be negative'

        return dim_xyz

    def gen_transf_matrices(self):
        '''
            For each video generate transformation matrices from camera
            coordinates to imu_gps coordinates and imu_gps transformation
            matrices from frame k to frame k-1.
        '''

        self.trans_camtoimu_matrices = {}
        self.trans_imutocam_matrices = {}
        self.oxts_projections = {}
        self.oxts_to0_projections = {}
        self.oxts_0tok_projections = {}
        self.delta_yaws = {}

        # calibration directory of training videos
        calib_dir = os.path.join(GlobalConfig.TRACKING_DATASET,
                                                  'calib',self.dataset_name)

        # get calibration files for each videos
        if python3:
            _, _, calib_files = next(os.walk(calib_dir))
        else:
            _, _,calib_files = os.walk(calib_dir).next()

        for f in calib_files:

            video_no = os.path.splitext(f)[0]

            # path to the calibration file
            path_calib = os.path.join(calib_dir, f)

            # transformation matrix from camera to the first imu coordinates
            self.trans_camtoimu_matrices[video_no] = cam_to_imu_transform(path_calib)

            self.trans_imutocam_matrices[video_no] = imu_to_cam_transform(path_calib)

            # transformation matrices from ith oxts coord to 1st oxts coord
            self.oxts_projections[video_no], self.delta_yaws[video_no] = oxts_prev_frame_trans(video_no=int(video_no), dataset_name = self.dataset_name)

            self.oxts_to0_projections[video_no], self.oxts_0tok_projections[video_no] = oxts_0_frame_trans(video_no=int(video_no), dataset_name = self.dataset_name)

def prepare_data(video_no, operation ='training'):
    '''
        Copy calib, plane, image_2, and velodyne files.
        In addition, write frame numbers in test.txt
    '''

    # kitti input path
    kitti_input_path = os.path.join(GlobalConfig.KITTI_DIR,'testing')
    # subdirectories under the kitti's input directory

    if python3:
        dirpath, folders, _ = next(os.walk(kitti_input_path))
    else:
        dirpath, folders, _ = os.walk(kitti_input_path).next()

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

            kitti_calib_path = os.path.join(GlobalConfig.TRACKING_DATASET,'calib',operation)
            src_calib_path = os.path.join(kitti_calib_path,video_no + '.txt')
            shutil.copy(src_calib_path, folder_path)
            kitti_tracking_utils.duplicate_calib()

        elif (f == 'planes'):

            kitti_tracking_utils.create_planes()

        elif(f == 'image_2'):

            kitti_img_2_path = os.path.join(GlobalConfig.TRACKING_DATASET,'image_2',operation)
            src_img_2_path = os.path.join(kitti_img_2_path,video_no)
            # copy images in src directory to the folder_path
            copy_tree(src_img_2_path, folder_path)

        elif(f == 'velodyne'):

            kitti_velodyne_path = os.path.join(GlobalConfig.TRACKING_DATASET,'velodyne',operation)
            src_velodyne_path = os.path.join(kitti_velodyne_path,video_no)
            # copy images in src directory to the folder_path
            copy_tree(src_velodyne_path, folder_path)

    # clean the test.txt and rewrite the frame numbers
    kitti_tracking_utils.rewrite_data_split_file()

    return True

def get_filenames(path_to_files):
    '''
       Find all files in a directory tree.
       Input: folders where the files reside under the
       Kitti directory, .i.e., '~/Kitti/folder1/folder2'.
       Return: The filenames.
    '''

    if python3:
        _, _, files = next(os.walk(path_to_files))
    else:
        _, _, files = os.walk(path_to_files).next()
    files = sorted(files)

    return files

def create_dataset_dir(folder_name):
    '''
        Create dataset directory to save numpy files.
        Return: directory where to save examples.
    '''

    # directory where dataset will be saved
    training_dataset_dir = os.path.join(GlobalConfig.ASSOCNET_DATASET,'training',folder_name)
    validation_dataset_dir = os.path.join(GlobalConfig.ASSOCNET_DATASET,'validation',folder_name)

    if not os.path.exists(training_dataset_dir):
        os.makedirs(training_dataset_dir)

    if not os.path.exists(validation_dataset_dir):
        os.makedirs(validation_dataset_dir)

    return training_dataset_dir, validation_dataset_dir

def get_obj_type_idx(obj_type):

    if (obj_type == object_types[2] or
        obj_type == object_types[3]):
        return 1

    else:
        return 0

if __name__ == '__main__':

    assocnet_dataset = KittiAssocnetDataset()
    assocnet_dataset.generate_dataset()
    assocnet_dataset.feat_ext.close_sessions()

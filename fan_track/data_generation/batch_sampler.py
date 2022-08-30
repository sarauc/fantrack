# -*- coding: utf-8 -*-

from fan_track.config.config import *
import os
import numpy as np
from fan_track.data_generation.map_generator import MapGenerator
from fan_track.data_generation.kitti_assocnet_dataset import KittiAssocnetDataset

class BatchSampler:

    def __init__(self):

        # list including pairs for training
        self.training = []
        # list including examples for validation
        self.validation = []

        # the number of examples in datasets
        self.lenT = 0
        self.lenV = 0

        # the ratio of validation to training
        self.ratio = 0.10

        # sample mean and variance of the training data set
        self.mu = None
        self.var = None

        # standardization counter
        self.std_cnt = 0

        self.init_paths()

    def load_dataset_files(self):
        '''
            Construct training and validation pairs by randomly
            selecting examples for each video. The ratio of the
            examples in validation set to those in training set
            must satisfy the ratio.
        '''

        moments_exist = os.path.isfile(self.training_dataset_path+'/moments.npy')

        for v in self.training_video_dir:

            # the path to the examples created for video v
            path_exp = self.training_dataset_path + '/{0}'.format(v)

            print('Processing training video:{0},'.format(v))

            # obtain the list of numpy files
            _, _, files = next(os.walk(path_exp))

            for f in files:

                # read the ith example file
                path_f = path_exp + '/{0}'.format(f)

                example = np.load(path_f, allow_pickle=True)

                self.training.append((v, f))
                targets = example.item()['targets']
                measurements = example.item()['measurements']

                if not moments_exist:
                    self.compute_standardization_params(targets)
                    self.compute_standardization_params(measurements)

        for v in self.validation_video_dir:
            # the path to the examples created for video v
            path_exp = self.validation_dataset_path + '/{0}'.format(v)

            print('Processing validation video:{0},'.format(v))

            # obtain the list of numpy files
            _, _, files = next(os.walk(path_exp))

            for f in files:
                self.validation.append((v, f))

        self.lenT = len(self.training)
        self.lenV = len(self.validation)

        # save moments used for standardization
        if not moments_exist:
            self.save_moments()


    def init_paths(self):
        '''
            Get path and directories to the dataset of pairs.
        '''

        # path to the npy pairs
        self.training_dataset_path = os.path.join(GlobalConfig.ASSOCNET_DATASET, 'training')

        self.validation_dataset_path = os.path.join(GlobalConfig.ASSOCNET_DATASET, 'validation')

        # # path where to save the moments
        self.path_moments = os.path.join(self.training_dataset_path,'moments')

        if not os.path.exists(GlobalConfig.ASSOCNET_DATASET):
            # generate bbox appearance dataset from kitti dataset
            dataset_obj = KittiAssocnetDataset()
            dataset_obj.generate_dataset()

        _, training_video_dir, _ =  next(os.walk(self.training_dataset_path))
        _, validation_video_dir, _ = next(os.walk(self.validation_dataset_path))

        self.training_video_dir = sorted(training_video_dir)
        self.validation_video_dir = sorted(validation_video_dir)

    def compute_standardization_params(self, objects):
        '''
            Compute the standardization parameters from training data.
            Standardization is computed for bbox params in the imu/gps
            coordinates at time k-1.
            :Arguments:
                Input: example is a dictionary including a pair of target
                       and measurement 3d bounding box parameters.
        '''

        for x in objects:

            self.std_cnt += 1

            if (self.std_cnt > 1):

                delta = x - self.mu
                self.mu += delta / self.std_cnt
                delta2 = x - self.mu
                self.M2 += delta * delta2
                self.var = self.M2 / (self.std_cnt -1)

            else:
                # sample mean and variance of the training data set
                self.mu = x
                self.var = None
                # M2 aggregates the squared distance from the mean
                self.M2 = 0

    def save_moments(self):
        '''
            Save the standardization parameters, i.e., the mean and
            the diagonal covariance matrix of the training data in a
            numpy file named as moments.
        '''

        # generate example pairs
        moments = {'mean':self.mu,'variance':self.var}

        np.save(self.path_moments,moments)

    def standardization(self,sample):

        if (self.mu is None or self.var is None):
            moments = np.load(self.path_moments + '.npy', allow_pickle=True)

            self.mu = moments.item()['mean']
            self.var = moments.item()['variance']

        return (sample - self.mu)/np.sqrt(self.var)

    def get_frame_pairs(self, is_training = True ,count = 1, startIndex = 0):
        '''
            Prepare inputs to the simnet.
            :Arguments:
                is_training: bool variable used to prepare training or validation dataset.
                count: is the number of pairs in each mini-batch, i.e., mini-batch size.
                startIndex: integer used to slice the dataset.
            :Return:
                startIndex: the new starting position for the dataset
                example: the training or validation input to the simnet.
        '''

        if (is_training):
            dataset = self.training
        else:
            dataset = self.validation

        set_size = len(dataset)
        examples = []

        if startIndex >= set_size:
            return -1, None

        for i, sample in enumerate(dataset[startIndex:]):

            if (i == count):
                break

            # the path to the ith examples created for video v

            if is_training:
                path_f = os.path.join(self.training_dataset_path,'{0}/{1}'.format(sample[0],sample[1]))
            else:
                path_f = os.path.join(self.validation_dataset_path, '{0}/{1}'.format(sample[0], sample[1]))
            # read the ith example file
            example = np.load(path_f, allow_pickle=True)

            # 3d position and shape params in the imu_gps coordinates at time k-1
            targets  = example.item()['targets']
            measurements = example.item()['measurements']

            # avod's features
            target_feat = example.item()['target_feat']
            meas_feat = example.item()['meas_feat']

            # list of 3d bbox params of targets and measurements
            mb_bbox = np.concatenate((targets, measurements), axis = 0)

            # standardization of 3d position and shape params
            mb_bbox= self.standardization(mb_bbox)

            # reshape mb_bbox from (2*mb_size,7) to (2*mb_size,1,7)
            mb_bbox = np.expand_dims(mb_bbox, axis = 1)

            # list of avod_fused features of targets and measurements
            mb_feat = np.concatenate((target_feat, meas_feat), axis = 0)

            mb_feat = np.asarray(mb_feat, dtype=np.float32)

            frame_pair = {'bbox': mb_bbox,
                          'feat': mb_feat,
                          'num_targets': example.item()['num_targets'],
                          'labels': example.item()['labels'],
                          'meas_locations': example.item()['meas_locations'],
                          'targets_xy': example.item()['targets_xy'],
                          'target_bboxes': example.item()['targets'],
                          'measurement_bboxes': example.item()['measurements'],
                          'target_avod_feat': example.item()['target_feat'],
                          'measurement_avod_feat': example.item()['meas_feat']
                         }

            examples.append(frame_pair)

        startIndex += count

        return startIndex, examples

    def shuffle_dataset(self, is_training):
        '''shuffle training or validation dataset'''

        if (is_training):
            dataset = self.training
        else:
            dataset = self.validation

        # suffle dataset
        np.random.shuffle(dataset)

if __name__ == '__main__':
    batch_sampler = BatchSampler()
    batch_sampler.load_dataset_files()
    batch_sampler.get_frame_pairs(is_training=True,count =1)

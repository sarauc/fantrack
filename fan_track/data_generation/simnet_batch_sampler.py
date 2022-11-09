import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from fan_track.data_generation.map_generator import MapGenerator
from random import shuffle
from fan_track.data_generation.kitti_simnet_dataset import KittiSimnetDataset
from fan_track.config.config import *
import tqdm

class SimnetBatchSampler(MapGenerator):

    def __init__(self, mb_size=128, mb_dir=None):

        super().__init__()

        # list including pairs (video,filename) for training
        self.training = []
        # list including examples for validation
        self.validation = []

        # the number of examples in datasets
        self.lenT = 0
        self.lenV = 0

        # the mini-batch size
        self.mb_size = mb_size

        # minimum and maximum of examples
        self.mins = None
        self.maxs = None

        # list of the mini batch files for training
        self.mb_files = []

        self.mbatch_count = 0

        # sample mean and variance of the training data set
        self.mu = None
        self.var = None

        # standardization counter
        self.std_cnt = 0
        self.path_to_dataset()

    def load_mb_filenames(self, mb_dir):
        if len(glob.glob(os.path.join(mb_dir,'training','*.npy'))) != 0:
            _, _, self.training_mb_files = next(os.walk(os.path.join(mb_dir,'training')))
            self.training_mbatch_count = len(self.training_mb_files)
        else:
            self.training_mb_files = []

        if len(glob.glob(os.path.join(mb_dir,'validation','*.npy'))) != 0:
            _, _, self.validation_mb_files = next(os.walk(os.path.join(mb_dir,'validation')))
            self.validation_mbatch_count = len(self.validation_mb_files)
        else:
            self.validation_mb_files = []


    def shuffle_minibatches(self, dataset_type):
        if dataset_type == 'training':
            shuffle(self.training_mb_files)
        else:
            shuffle(self.validation_mb_files)

    def path_to_dataset(self):
        '''
            Get path and directories to the dataset of pairs.
        '''

        # path to the npy pairs
        self.dataset_path = GlobalConfig.SIMNET_DATASET_PATH

        # path to the validation list
        self.valid_list_path = os.path.join(self.dataset_path, 'val_list.npy')

        # mini batch directory
        self.mb_dir = GlobalConfig.SIMNET_MINIBATCH_PATH

        if not (os.path.exists(self.mb_dir + '/training')):
            os.makedirs(self.mb_dir+ '/training')

        if not (os.path.exists(self.mb_dir + '/validation')):
            os.makedirs(self.mb_dir + '/validation')

        # path where to save the moments
        self.path_moments = os.path.join(self.dataset_path, 'moments')

        # check if dataset_path exists and it is a directory
        if True:
            # generate bbox appearance dataset from kitti dataset
            dataset_obj = KittiSimnetDataset()

            dataset_obj.generate_dataset()

        _, video_dir, _ = next(os.walk(self.dataset_path))

        self.video_dir = sorted(video_dir)

    def construct(self):
        '''
            Construct training and validation pairs by randomly
            selecting examples for each video. The ratio of the
            examples in validation set to those in training set
            must satisfy the ratio.
        '''

        print('Computing Standardization Params..')

        if os.path.exists(os.path.join(GlobalConfig.SIMNET_DATASET_PATH, 'moments.npy')):
            compute_std_params = False
        else:
            compute_std_params = True

        for video in tqdm.tqdm(self.video_dir):

            training_path = os.path.join(self.dataset_path,video,'training')
            validation_path = os.path.join(self.dataset_path,video,'validation')

            # obtain the list of numpy files
            _, _, training_files = next(os.walk(training_path))
            _, _, validation_files = next(os.walk(validation_path))

            # place the examples in training and validation datasets
            for training_file in training_files:

                if compute_std_params:
                    training_ex = np.load(os.path.join(training_path,training_file),allow_pickle=True)

                self.training.append((video, training_file))

                if compute_std_params:
                    self.compute_std_params(training_ex)

            for i, validation_file in enumerate(validation_files):
                self.validation.append((video,validation_file))

        self.lenT = len(self.training)
        self.lenV = len(self.validation)

        if compute_std_params:
            # save moments used for standardization
            self.save_moments()

    def save_validation_list(self):
        '''Save the list of validation examples in a numpy file'''

        np.save(self.valid_list_path, self.validation)

    def compute_std_params(self, example):
        '''
            Compute the standardization parameters from training data.
            Standardization is computed for bbox params in the imu/gps
            coordinates at time k-1.
            Input: example is a dictionary including a pair of target
                   and measurement 3d bounding box parameters.
        '''
        keys = ['target', 'measurement']

        for k in keys:

            x = example.item()[k]

            self.std_cnt += 1

            if (self.std_cnt > 1):

                delta = x - self.mu
                self.mu += delta / self.std_cnt
                delta2 = x - self.mu
                self.M2 += delta * delta2
                self.var = self.M2 / (self.std_cnt - 1)

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
        moments = {'mean': self.mu, 'variance': self.var}

        np.save(self.path_moments, moments)

    def standardization(self, sample):

        if (self.mu is None or self.var is None):
            moments = np.load(self.path_moments + '.npy', allow_pickle=True)

            self.mu = moments.item()['mean']
            self.var = moments.item()['variance']

        return (sample - self.mu) / np.sqrt(self.var)

    def generate_and_save_mini_batch(self, file_mappings, dataset_type):

        mb_idx = 0

        shuffle(file_mappings)
        dataset_size = len(file_mappings)

        bar = tqdm.tqdm(total=int(dataset_size/self.mb_size))
        print('Preparing mini-batches...')

        while (True):
            # first and last indices of the mini-batch examples
            first = mb_idx * self.mb_size
            last = first + self.mb_size

            if(first >= dataset_size):
                break

            if(last>dataset_size):
                last = dataset_size

            mb_filenames = file_mappings[first:last]

            # length of mini batch
            len_mb = last - first

            # targets and measurements in the imu_gps coordinates at time k-1
            # where k-1 is the targets' time and k is the measurements' time
            targets = [None] * len_mb
            measurements = [None] * len_mb

            # 3d positions of measurements in the imu_gps coordinates at time k
            meas_imu_k = [None] * len_mb

            labels = [None] * len_mb
            target_feat = [None] * len_mb
            meas_feat = [None] * len_mb

            for i, metadatum in enumerate(mb_filenames):
                # the path to the ith examples created for video v
                path_f = os.path.join(self.dataset_path, metadatum[0], dataset_type, metadatum[1])

                # read the ith example file
                example = np.load(path_f, allow_pickle=True)

                # 3d position and shape params in the imu_gps coordinates at time k-1
                targets[i] = example.item()['target']
                measurements[i] = example.item()['measurement']

                # 3d position of the measurements in the imu_gps coordinates at time k
                meas_imu_k[i] = example.item()['meas_imu_k']

                # avod's features
                target_feat[i] = example.item()['target_feat']
                meas_feat[i] = example.item()['meas_feat']

                labels[i] = example.item()['simnet_label']

            # list of 3d bbox params of targets and measurements
            tar_meas = targets + measurements

            # mini-batch 3d bbox params
            mb_bbox = np.asarray(tar_meas, dtype=np.float32)

            # locate measurements and compute coordinates of local maps
            self.loc_ind_coords(np.asarray(meas_imu_k))

            # standardization of 3d position and shape params
            mb_bbox = self.standardization(mb_bbox)

            targets_bbox = mb_bbox[0:len(targets),:]
            measurements_bbox = mb_bbox[len(targets):,:]

            targets_bbox = np.expand_dims(targets_bbox, axis=1)
            measurements_bbox = np.expand_dims(measurements_bbox, axis=1)

            # reshape mb_bbox from (2*mb_size,7) to (2*mb_size,1,7)
            mb_bbox = np.expand_dims(mb_bbox, axis=1)

            # list of avod_fused features of targets and measurements
            mb_feat = target_feat + meas_feat

            mb_feat = np.asarray(mb_feat, dtype=np.float32)

            target_feat = mb_feat[0:len(targets),:]
            meas_feat =  mb_feat[len(targets):,:]

            # mini-batch labels
            mb_labels = np.asarray(labels, dtype=np.float32)

            # add label axis to reshape mb_labels to [len_mb,1]
            mb_labels = np.expand_dims(mb_labels, axis=1)

            '''save mini-batch'''
            # generate example
            mini_batch = {'bbox': mb_bbox,
                          'feat': mb_feat,
                          'tar_bbox': targets_bbox,
                          'meas_bbox': measurements_bbox,
                          'tar_feat': target_feat,
                          'meas_feat': meas_feat,
                          'labels': mb_labels
                          }

            filename = str(mb_idx).rjust(7, '0') + '.npy'
            mb_path = os.path.join(self.mb_dir, dataset_type, filename)
            np.save(mb_path, mini_batch)

            # next mini-batch index
            bar.update(1)
            mb_idx += 1
        bar.close()
        print('\n')

    def next_mb(self, mb_idx, operation, mb_dir):
        '''
            Read a mini-batch for training of the network.
            Input: mb_idx is the index of the mini-batch.
                   mb_dir is the min-batch directory.
        '''

        if operation == 'training':
            mb_files = self.training_mb_files
        else:
            mb_files = self.validation_mb_files

        self.mbatch_count = len(mb_files)
        if (mb_idx < self.mbatch_count):

            mb_file = mb_files[mb_idx]
            mb_path = os.path.join(mb_dir, operation, mb_file)
            mb = np.load(mb_path, allow_pickle=True)
            mb_idx += 1

            return mb_idx, mb
        else:
            return -1, None

def main():
    batch_sampler = SimnetBatchSampler(mb_dir = GlobalConfig.SIMNET_MINIBATCH_PATH)
    batch_sampler.construct()
    batch_sampler.generate_and_save_mini_batch(file_mappings = batch_sampler.training,
                                               dataset_type = 'training')
    batch_sampler.generate_and_save_mini_batch(file_mappings=batch_sampler.validation,
                                               dataset_type='validation')

if __name__ == '__main__':
    main()

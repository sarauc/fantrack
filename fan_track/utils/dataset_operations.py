import numpy as np
import tensorflow as tf
import fan_track.utils
import matplotlib.pyplot as plt
import os
import gc
import time
import shutil
import random

import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core import box_3d_encoder
from avod.core import box_3d_projector
from avod.core.models.avod_model import AvodModel
from avod.core.models.rpn_model import RpnModel
from fan_track.config.config import GlobalConfig
from wavedata.tools.core import calib_utils
from wavedata.tools.visualization import vis_utils

class DatasetConverter:

    def __init__(self, operation = 'training'):

        self.sequence_name = []
        self.training_split = 0.80
        self.current_frame_number = 0
        self.operation = operation
        self.node_dir = os.path.dirname(os.path.realpath(__file__))
        self.root_dir = os.path.split(self.node_dir)[0]

        # Source paths
        self.image_source_path = GlobalConfig.TRACKING_DATASET + '/image_2/' + self.operation
        self.velodyne_source_path = GlobalConfig.TRACKING_DATASET + '/velodyne/' + self.operation
        self.labels_source_path = GlobalConfig.TRACKING_DATASET + '/labels/label_02'
        self.calib_source_path = GlobalConfig.TRACKING_DATASET + '/calib/' + self.operation

        # Destination paths
        self.image_dest_path = GlobalConfig.KITTI_DIR + '/training/image_2'
        self.velodyne_dest_path = GlobalConfig.KITTI_DIR + '/training/velodyne'
        self.labels_dest_path = GlobalConfig.KITTI_DIR + '/training/label_2'
        self.planes_dest_path = GlobalConfig.KITTI_DIR + '/training/planes'
        self.calib_dest_path = GlobalConfig.KITTI_DIR + '/training/calib'

    def set_source_path(self, dataset = 'tracking'):

        if dataset == 'tracking':
            self.image_source_path = GlobalConfig.TRACKING_DATASET + '/image_2/' + self.operation
            self.velodyne_source_path = GlobalConfig.TRACKING_DATASET + '/velodyne/' + self.operation
            self.labels_source_path = GlobalConfig.TRACKING_DATASET + '/labels/label_02'
            self.calib_source_path = GlobalConfig.TRACKING_DATASET + '/calib/' + self.operation
        else:
            self.image_source_path = os.path.join(GlobalConfig.OBJECT_DETECTION_DATASET,'image_2')
            self.velodyne_source_path = os.path.join(GlobalConfig.OBJECT_DETECTION_DATASET, 'velodyne')
            self.labels_source_path = os.path.join(GlobalConfig.OBJECT_DETECTION_DATASET, 'label_2')
            self.calib_source_path = os.path.join(GlobalConfig.OBJECT_DETECTION_DATASET, 'calib')

    def copy_data(self):

        train_file = open(os.path.join(GlobalConfig.KITTI_DIR, 'train.txt'), 'w')
        val_file = open(os.path.join(GlobalConfig.KITTI_DIR, 'val.txt'), 'w')
        train_val_file = open(os.path.join(GlobalConfig.KITTI_DIR, 'trainval.txt'), 'w')
        # test_file = open(os.path.join(GlobalConfig.KITTI_DIR, 'test.txt'))

        # Prepare source detection dataset
        _, _, det_img_files = next(os.walk(os.path.join(GlobalConfig.OBJECT_DETECTION_DATASET, 'image_2')))
        src_det_indices = [int(ch[0:6]) for ch in det_img_files]

        # Delete existing data
        # subdirectories under the kitti's input directory
        dirpath, folders, _ = next(os.walk(os.path.join(GlobalConfig.KITTI_DIR, self.operation)))

        # Clean the folders
        for f in folders:
            folder_path = os.path.join(dirpath, f)

            files = os.listdir(folder_path)

            # delete files under the folder
            [os.remove(folder_path + '/' + f) for f in files]

        video_dict = {}
        randomized_tracking_frame_numbers = []
        with open('tracking_to_det.seqmap', "r") as fh:
            for i, l in enumerate(fh):
                fields = l.split(" ")
                video_no = "%04d" % int(fields[0])

                start_frame = int(fields[2])
                total_frames = int(fields[3])
                last_training_frame = int(total_frames*self.training_split)

                label_dict = {}
                # load labels for the video
                with open(self.labels_source_path + '/' + video_no +'.txt',"r") as label_h:
                    for i, label_line in enumerate(label_h):
                        label_fields = label_line.split(" ")
                        label_frame_number = int(label_fields[0])

                        detection_label_line = ''
                        for label_index in range(2,17):

                            detection_label_word = label_fields[label_index]

                            # Handling conversion of truncation from enum to float
                            if label_index == 3:
                                if label_fields[label_index] == '0':
                                    detection_label_word = '0.0'
                                elif label_fields[label_index] == '1':
                                    detection_label_word = '0.25'
                                elif label_fields[label_index] == '2':
                                    detection_label_word = '0.45'

                            if detection_label_line == '':
                                detection_label_line = detection_label_word
                            else:
                                detection_label_line = detection_label_line + ' ' + detection_label_word

                        if label_frame_number in label_dict:
                            label_dict[label_frame_number].append(detection_label_line)
                        else:
                            label_dict[label_frame_number] = [detection_label_line]

                    label_h.close()

                video_dict[video_no + 'label_dict'] = label_dict
                video_dict[video_no + 'start_frame'] = start_frame
                video_dict[video_no + 'total_frames'] = total_frames

                randomized_frame_numbers = random.sample(range(start_frame, start_frame + total_frames), total_frames)
                randomized_tracking_frame_numbers.extend([ video_no+"%06d" % fr  for fr in randomized_frame_numbers])

            fh.close()

        # Interleave object detection frame and tracking frame
        while(True):

            # Decide whether to choose detection or tracking dataset for the iteration
            if self.current_frame_number %2 == 0 and len(src_det_indices)!=0:
                self.set_source_path(dataset='detection')
                source_frame = "%06d" % src_det_indices.pop(0)

                # Copy Labels
                label_filename = "%06d" % self.current_frame_number + '.txt'
                shutil.copy(os.path.join(self.labels_source_path,source_frame + '.txt'), os.path.join(self.labels_dest_path,label_filename))

                # Copy the image
                img_filename = "%06d" % self.current_frame_number + '.png'
                shutil.copy(os.path.join(self.image_source_path,source_frame + '.png'), os.path.join(self.image_dest_path,img_filename))

                # Copy Velodyne
                velodyne_filename = "%06d" % self.current_frame_number + '.bin'
                shutil.copy(os.path.join(self.velodyne_source_path,source_frame + '.bin'),os.path.join(self.velodyne_dest_path,velodyne_filename))

                # Copy calib
                calib_filename = "%06d" % self.current_frame_number + '.txt'
                shutil.copy(os.path.join(self.calib_source_path,source_frame + '.txt'), os.path.join(self.calib_dest_path,calib_filename))

                train_file.write("%06d" % self.current_frame_number + '\n')

            elif len(randomized_tracking_frame_numbers)!=0:
                self.set_source_path(dataset='tracking')
                randomized_tracking_frame = randomized_tracking_frame_numbers.pop(0)
                video_no = randomized_tracking_frame[0:4]
                source_frame = randomized_tracking_frame[4:10]
                index = int(source_frame)
                label_dict = video_dict[video_no + 'label_dict']
                total_frames = video_dict[video_no + 'total_frames']
                last_training_frame = int(total_frames*self.training_split)

                # write Labels
                label_write_h = open(self.labels_dest_path + '/' + "%06d" % self.current_frame_number + '.txt', "w")
                if index in label_dict:
                    for label_line in label_dict[index]:
                        label_write_h.write(label_line)
                else:
                    continue

                # Copy the image
                img_filename = "%06d" % self.current_frame_number + '.png'
                shutil.copy(self.image_source_path + '/' + video_no + '/' + source_frame + '.png',
                            self.image_dest_path + '/' + img_filename)

                # Copy Velodyne
                velodyne_filename = "%06d" % self.current_frame_number + '.bin'
                shutil.copy(self.velodyne_source_path + '/' + video_no + '/' + source_frame + '.bin',
                            self.velodyne_dest_path + '/' + velodyne_filename)

                # Copy calib
                calib_filename = "%06d" % self.current_frame_number + '.txt'
                shutil.copy(self.calib_source_path + '/' + video_no + '.txt',
                            self.calib_dest_path + '/' + calib_filename)

                label_write_h.close()

                if index <= last_training_frame:
                    train_file.write("%06d" % self.current_frame_number + '\n')
                else:
                    val_file.write("%06d" % self.current_frame_number + '\n')
            else:
                break

            # Common updates
            train_val_file.write("%06d" % self.current_frame_number + '\n')
            # train_file.write("%06d" % self.current_frame_number + '\n')
            self.current_frame_number+=1
            print(self.current_frame_number)

        train_val_file.close()
        train_file.close()
        val_file.close()

        # Copy planes
        self.create_planes()

    def create_planes(self):
        '''create the ground plane as y = 0*sx + 0*sy + h where sx, sz are slopes and h is the height of the sensor'''

        global num_frames
        global frames

        _, _, frames = next(os.walk(self.velodyne_dest_path))

        num_frames = len(frames)

        # ground plane in the format of [ground_normal,height of the sensor] where normal is facing up
        ground_plane = [0, -1, 0, 1.73]

        for file_num in range(num_frames):

            filename = '{:06}'.format(file_num) + '.txt'

            with open(self.planes_dest_path + '/' + filename, 'w+') as f:

                for i in range(4):
                    if (i == 0):
                        f.write('# Matrix\n')
                    elif (i == 1):
                        f.write('WIDTH 4\n')
                    elif (i == 2):
                        f.write('HEIGHT 1\n')
                    else:
                        f.write(''.join('{:.6e}'.format(elem) + ' ' for elem in ground_plane))

    def create_data_splits_files(self):

        train_file = open(os.path.join(GlobalConfig.KITTI_DIR,'train.txt'),'w')
        val_file = open(os.path.join(GlobalConfig.KITTI_DIR, 'val.txt'),'w')
        train_val_file = open(os.path.join(GlobalConfig.KITTI_DIR, 'trainval.txt'),'w')
        # test_file = open(os.path.join(GlobalConfig.KITTI_DIR, 'test.txt'))

        training_split = round(self.training_split*self.current_frame_number)

        for index in range(self.current_frame_number):

            train_val_file.write("%06d" % index + '\n')

            if index < training_split:
                train_file.write("%06d" % index + '\n')
            else:
                val_file.write("%06d" % index + '\n')

    def create_labels(self, dataset_subtype = 'train'):
        _, _, videos = next(os.walk(self.labels_source_path))

        # Identify the frame ranges
        seq_dict = {}
        with open('tracking_to_det.seqmap', "r") as fh:
            for i, l in enumerate(fh):
                fields = l.split(" ")
                video_no = "%04d" % int(fields[0])
                vid_dict = {}

                start_frame = int(fields[2])
                total_frames = int(fields[3])
                last_training_frame = int(total_frames * self.training_split)

                seq_dict[video_no + '.txt_start']=start_frame
                seq_dict[video_no + '.txt_total']=total_frames
                seq_dict[video_no + '.txt_last_training_frame']=last_training_frame

        # Create Label files
        for video in videos:

            with open(os.path.join(self.labels_source_path,video), 'r') as f:
                open(self.root_dir+'/evaluation/python/data/tracking/label_02/' + video, 'w').close()
                for i, l in enumerate(f):
                    fields = l.split(" ")

                    if dataset_subtype == 'train':
                        if int(fields[0]) <= seq_dict[video + '_last_training_frame']:
                            writefile = open(self.root_dir+'/evaluation/python/data/tracking/label_02/' + video,'a')
                            fields[0] = str(int(fields[0]))

                            firstfield = True
                            for field in fields:
                                if firstfield:
                                    writefile.write(field)
                                    firstfield = False
                                else:
                                    writefile.write(' ' + field)

                    else:
                        if int(fields[0]) > seq_dict[video + '_last_training_frame']:
                            writefile = open(self.root_dir+'/evaluation/python/data/tracking/label_02/'+video,'a')
                            fields[0] = str(int(fields[0]) - (seq_dict[video + '_last_training_frame']+1))

                            firstfield = True
                            for field in fields:
                                if firstfield:
                                    writefile.write(field)
                                    firstfield = False
                                else:
                                    writefile.write(' ' + field)

            writefile.close()

        if dataset_subtype == 'train':
            seqfile_name = self.root_dir+'/evaluation/python/data/tracking/evaluate_tracking.seqmap.training'
        else:
            seqfile_name = self.root_dir+'/evaluation/python/data/tracking/evaluate_tracking.seqmap.validation'

        # Create seqmap file
        with open(seqfile_name, "w") as fh:

            for video in range(0,21):
               video_no = "%04d" % video

               if dataset_subtype == 'train':
                   seqmap_contents = video_no + ' ' + 'empty ' + "%06d" % seq_dict[
                       video_no + '.txt_start'] + ' ' + "%06d" % seq_dict[video_no + '.txt_last_training_frame'] + '\n'
               else:
                   seqmap_contents = video_no + ' ' + 'empty ' + "%06d" % seq_dict[
                       video_no + '.txt_last_training_frame'] + ' ' + "%06d" % seq_dict[video_no + '.txt_total'] + '\n'

               fh.write(seqmap_contents)

        fh.close()

if __name__ == '__main__':

    converter = DatasetConverter()

    # converter.copy_data()
    converter.create_labels(dataset_subtype='validation')

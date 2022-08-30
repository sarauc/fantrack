import os
import numpy as np
from fan_track.utils import cam_to_imu_transform
import random as rand
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fan_track.utils import read_calib_matrix
from fan_track.utils import project_to_image_space
from fan_track.utils import draw, read_img
from scipy import ndimage
import matplotlib


class InputData:

    def __init__(self, gt, map_length=80, map_width=80, length_resolution=1,
                 width_resolution=1,  max_target=21, max_obs=21, calib_path_in="", args=None):
        """
           Constructor, initializes the object given the parameters.
        """
        self.gt = gt
        self.width = map_width
        self.length = map_length
        self.wr = width_resolution
        self.lr = length_resolution
        self.max_target = max_target
        self.max_meas = max_obs
        self.calib_path = calib_path_in
        self.args = args

        # number of pixels along the X axis
        self.num_pix_x = int(self.length / self.lr)
        # number of pixels along the Y axis
        self.num_pix_y = int(self.width / self.wr)
        # crop region
        self.crop_size = int(20 / self.lr)

        self.cropped_maps = []
        self.target_no = []
        self.meas_no = []
        self.labels = []

    def generation(self, first=0, last=21):
        """
           Generate the data
           Inputs: Class member variables
           Outputs: data for each pair of consecutive frame for all 21 KITTI training sequences
        """
        # Get two consecutive frames
        for seq_idx in range(first, last):
            seq_gt = self.gt.groundtruth[seq_idx]
            calib_file = os.path.join(self.calib_path, "%04d.txt" % seq_idx)
            t_cam_to_imu = cam_to_imu_transform(calib_file)

            center_meas_save = []
            bbox_meas_save = []
            meas_id_save = []
            bbox_meas_camera_save = []

            for f_idx in range(1, len(seq_gt) - 1):
                print("Reading Sequence: %d \t \t Frame: %d" % (seq_idx, f_idx))
                # Read the data present in the label file
                center_target, center_meas, tar_id, meas_id, bbox_tar, bbox_meas, bbox_tar_camera, bbox_meas_camera = \
                    self.get_frame_pair_data(seq_gt, f_idx, t_cam_to_imu, center_meas_save, bbox_meas_save, meas_id_save, bbox_meas_camera_save)
                # Save to use next time as target
                center_meas_save = center_meas
                bbox_meas_save = bbox_meas
                meas_id_save = meas_id
                bbox_meas_camera_save = bbox_meas_camera

                # Get the processed data for targets and measurements for a pair of frames
                flag = self.get_io_maps(center_target, center_meas, bbox_tar, bbox_meas, tar_id, meas_id, f_idx, seq_idx)
                if flag:
                    self.target_no.append(len(center_target))
                    self.meas_no.append(len(center_meas))

                # Visualization
                self.visualize_global(center_target, center_meas, f_idx, seq_idx, calib_file, bbox_tar_camera)

    def iterate_over_bbox(self, gt_f, t_cam_to_imu):
        """
           Loop through all bounding boxes in a frame
        """
        bbox_cam = []
        b_boxes = []
        ids = []
        # Bounding boxes for targets in frame f_idx for sequence seq_idx
        for itr in range(len(gt_f)):
            bbox_ry_cam = [gt_f[itr].X, gt_f[itr].Y, gt_f[itr].Z, gt_f[itr].l, gt_f[itr].w, gt_f[itr].h, gt_f[itr].yaw]
            bbox_pos_imu = np.dot(t_cam_to_imu, np.append(bbox_ry_cam[0:3], 1))
            # Check if IMU co-ordinates within designated area
            if bbox_pos_imu[0] < self.length and np.abs(bbox_pos_imu[1]) < self.width / 2:
                b_boxes.append(np.array(bbox_pos_imu))
                ids.append(gt_f[itr].track_id)
                bbox_cam.append(bbox_ry_cam)
        return b_boxes, ids, bbox_cam

    def get_frame_pair_data(self, seq_gt, f_idx, t_cam_to_imu, center_meas_save, bbox_meas_save, meas_id_save, bbox_meas_camera_save):
        """
           Get input and labels for consecutive frame pairs
        """
        # Only if first frame, build both targets and measurements
        if f_idx == 1:
            bbox_tar, tar_id, bbox_tar_camera = self.iterate_over_bbox(seq_gt[f_idx - 1], t_cam_to_imu)
            # Target coordinates on the map (X,Y)
            center_target = self.mapping(bbox_tar)

            bbox_meas, meas_id, bbox_meas_camera = self.iterate_over_bbox(seq_gt[f_idx], t_cam_to_imu)
            # Measurement coordinates on the map (X,Y)
            center_meas = self.mapping(bbox_meas)

        else:
            # Otherwise use previous measurements as current targets
            bbox_tar = bbox_meas_save
            center_target = center_meas_save
            tar_id = meas_id_save
            bbox_tar_camera = bbox_meas_camera_save

            bbox_meas, meas_id, bbox_meas_camera = self.iterate_over_bbox(seq_gt[f_idx], t_cam_to_imu)
            # Measurement coordinates on the map (X,Y)
            center_meas = self.mapping(bbox_meas)

        return center_target, center_meas, tar_id, meas_id, bbox_tar, bbox_meas, bbox_tar_camera, bbox_meas_camera

    def mapping(self, bbox_3d):
        """
            Compute x-y locations of objects in a global map
            Inputs: bbox_3d
            Outputs: one iterator: the list of two tuples, each of which is the
                     location of objects on a global map. Returns am empty list
                     if map had no bounding boxes to locate.
        """
        # x-y grid coordinates
        bbox_3d = np.array(bbox_3d)
        if len(bbox_3d) > 0:
            x_idx = list(map(lambda x: int(x / self.lr), bbox_3d[:, 0]))
            y_idx = list(map(lambda y: int((y + self.width/2)/self.wr), bbox_3d[:, 1]))
            return list(zip(x_idx, y_idx))
        else:
            return []

    def get_io_maps(self, center_target, center_meas, bbox_tar, bbox_meas, tar_id, meas_id, f_idx, seq_idx):
        """
           Get the X-input data and training labels for the network
        """
        tar_map = np.zeros([self.crop_size + 1, self.crop_size + 1, self.max_target])
        label_map = np.zeros([self.crop_size + 1, self.crop_size + 1, self.max_target])
        tar_total = len(center_target)
        meas_total = len(center_meas)

        if tar_total > 0 and meas_total > 0:
            for tar_no in range(tar_total):

                for meas_no in range(meas_total):
                    # Check if object in range
                    vert_dist = center_target[tar_no][0] - center_meas[meas_no][0]
                    horz_dist = center_target[tar_no][1] - center_meas[meas_no][1]

                    if np.abs(vert_dist) < self.crop_size / 2 and np.abs(horz_dist) < self.crop_size / 2:
                        tar_idx = int(self.crop_size/2 + 1)
                        # Euclidean distance of every target from meas j
                        x_norm = (bbox_tar[tar_no][0] - bbox_meas[meas_no][0]) / self.args.map_length
                        y_norm = (bbox_tar[tar_no][1] - bbox_meas[meas_no][1]) / self.args.map_width
                        z_norm = (bbox_tar[tar_no][2] - bbox_meas[meas_no][2]) / 5
                        tar_map[self.crop_size - (tar_idx - horz_dist), tar_idx - vert_dist, tar_no] = np.linalg.norm([x_norm, y_norm, z_norm])
                        # Get labels
                        if tar_id[tar_no] == meas_id[meas_no]:
                            label_map[self.crop_size - (tar_idx - horz_dist), tar_idx - vert_dist, tar_no] = 1
                        # Visualize the map for each target
                        self.visualize_local(tar_id[tar_no], tar_map[:, :, tar_no], label_map[:, :, tar_no], f_idx, seq_idx)

            self.cropped_maps.append(tar_map)
            self.labels.append(label_map)
            flag = 1
        else:
            flag = 0
        return flag

    def visualize_local(self, tar_id, tar_map, label_map, f_idx, seq_idx):
        matplotlib.image.imsave('IO_visualization/%d/frame_%04d_tar_%02d.png' %(seq_idx, f_idx, tar_id), tar_map)
        matplotlib.image.imsave('IO_visualization/%d/frame_%04d_label_%02d.png' % (seq_idx, f_idx, tar_id), label_map)

    def visualize_global(self, center_target, center_meas, f_idx, seq_idx, calib_file, bbox_tar_camera):
        """
           Visualize the image, targets and measurements on the grid
        """
        max_tar = len(center_target)
        max_meas = len(center_meas)
        global_map = np.zeros((self.num_pix_x, self.num_pix_y, 3), dtype=np.uint8)
        # Iterator for targets and measurements
        n_colors_t = 0
        n_colors_m = 0
        # Create 2x2 sub plots
        gs = gridspec.GridSpec(2, 2)
        plt.figure()
        ax3 = plt.subplot(gs[0, :])  # row 0, span all columns
        ax1 = plt.subplot(gs[1, 0])  # row 1, col 0
        ax2 = plt.subplot(gs[1, 1])  # row 1, col 1

        # Generate image being worked upon
        img_path = os.path.join("/home/p6bhatta/WaterlooResearch/Datasets/kitti_tracking/data_tracking_image_2/" \
                                "training/image_02/", "%04d/%06d.png") % (seq_idx, f_idx)
        calib_p2 = read_calib_matrix(calib_file)
        img = read_img(img_path)
        for cc, t in enumerate(bbox_tar_camera):
            x = project_to_image_space(np.array(t), calib_p2)
            draw(x, img, cc)
        ax3.imshow(img)
        ax3.axis('off')

        # generate different colors to map targets
        while n_colors_t < max_tar:
            # sample a random color
            c = rand.sample(range(50, 255), 3)
            # map targets
            c_idx = center_target[n_colors_t]
            try:
                global_map[c_idx[0], c_idx[1], :] = c
            except IndexError:
                print('target is too far from the car')
            n_colors_t += 1
        t_map = global_map
        ax1.imshow(ndimage.rotate(np.fliplr(t_map), 90), origin='lower')
        ax1.grid(True, color='white', linewidth=0.2)
        ax1.set_title('Targets')
        # make every nth tick's label visible
        rate_vis = 10
        plt.sca(ax1)
        plt.yticks(np.arange(0, int(self.num_pix_y), 1), np.arange(-int(self.num_pix_y * self.wr / 2), int(self.num_pix_y * self.wr / 2), self.wr))
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels()[::rate_vis], visible=True)
        plt.xticks(np.arange(0, int(self.num_pix_x), 1), np.arange(0, int(self.num_pix_x * self.lr), self.lr))
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_xticklabels()[::rate_vis], visible=True)

        # clear the global map
        global_map.fill(0)
        while n_colors_m < max_meas:
            # sample a random color
            c = rand.sample(range(50, 255), 3)
            # map measurements
            c_idx = center_meas[n_colors_m]
            try:
                global_map[c_idx[0], c_idx[1], :] = c
            except IndexError:
                print('measurement is too far from the car')
            n_colors_m += 1
        m_map = global_map
        ax2.imshow(ndimage.rotate(np.fliplr(m_map), 90), origin='lower')
        ax2.grid(True, color='white', linewidth=0.2)
        ax2.set_title('Measurements')
        # make every nth tick's label visible
        rate_vis = 10
        plt.sca(ax2)
        plt.yticks(np.arange(0, int(self.num_pix_y), 1),
                   np.arange(-int(self.num_pix_y * self.wr / 2), int(self.num_pix_y * self.wr / 2), self.wr))
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels()[::rate_vis], visible=True)
        plt.xticks(np.arange(0, int(self.num_pix_x), 1), np.arange(0, int(self.num_pix_x * self.lr), self.lr))
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels()[::rate_vis], visible=True)

        # Generate the actual plot
        plt.suptitle('\n Frame %d \n Sequence %d \n' % (f_idx, seq_idx))
        plt.savefig('IO_visualization/%d/frame_%04d.png' %(seq_idx, f_idx))
        plt.close()


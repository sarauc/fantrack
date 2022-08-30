from _functools import reduce
import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
from sklearn.preprocessing import minmax_scale
from fan_track.config.config import GlobalConfig
from scipy.special import softmax

class PositionBased:
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 44
    config.inter_op_parallelism_threads = 44  # High performance
    graph = tf.Graph()

    def __init__(self, args):
        self.args = args
        self.pos_shp_shape = [None, 1, 7]
        self.cntr_xy_shape = [None, 1, 2]
        self.loc_shape = [reduce((lambda h, w: h * w), (args.map_length, args.map_width))]
        self.corr_map_shape = [1, GlobalConfig.CROP_SIZE, GlobalConfig.CROP_SIZE, self.args.max_targets]
        self.map_shape = list((args.map_length, args.map_width))
        self.similarity_distance = 'euclidean'

    def get_image_dist(self, dist_metric, target, measurement, image_path):

        image = cv2.imread(image_path)
        t_x1 = int(target.object_label.x1)
        t_x2 = int(target.object_label.x2)
        t_y1 = int(target.object_label.y1)
        t_y2 = int(target.object_label.y2)

        m_x1 = int(measurement.object_label.x1)
        m_x2 = int(measurement.object_label.x2)
        m_y1 = int(measurement.object_label.y1)
        m_y2 = int(measurement.object_label.y2)

        cropped_target = image[t_y1:t_y2,t_x1:t_x2]
        cropped_measurement = image[m_y1:m_y2,m_x1:m_x2]

        target_hist = cv2.calcHist([cropped_target], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        target_hist = cv2.normalize(target_hist, target_hist).flatten()

        measurement_hist = cv2.calcHist([cropped_measurement], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        measurement_hist = cv2.normalize(measurement_hist, measurement_hist).flatten()

        if dist_metric == 'IMAGE_BHATTACHARYYA':
            similarity = cv2.compareHist(target_hist, measurement_hist, cv2.HISTCMP_BHATTACHARYYA)

        if dist_metric == 'Chi-Squared':
            similarity = cv2.compareHist(target_hist, measurement_hist, cv2.HISTCMP_CHISQR)

        return similarity

    def euclidean_similarity(self, combined_bboxes, num_targets, measurement_locations, target_centers, targets, measurements, image_path):

        target_poses = combined_bboxes[0:num_targets,0:3]
        measurement_poses = combined_bboxes[num_targets:,0:3]
        mapped_meas_full = np.zeros([measurement_locations.shape[0],num_targets])

        for tar_index in range(num_targets):

            temp_array = []
            target = targets[tar_index]
            target_pose = target_poses[tar_index]
            target_pose = np.reshape(target_pose, [1, -1])

            for idx,item in enumerate(measurement_locations):
                measurement_pose = measurement_poses[int(item)]
                measurement_pose = np.reshape(measurement_pose,[1,-1])
                measurement = measurements[int(item)]
                if item != -1:

                    if self.similarity_distance == 'euclidean':
                        mapped_meas_full[idx][tar_index] = np.linalg.norm(target_pose-measurement_pose)
                    elif self.similarity_distance == 'manhattan':
                        mapped_meas_full[idx][tar_index] = cdist(target_pose,measurement_pose, metric='cityblock')
                    elif self.similarity_distance == 'correlation':
                        mapped_meas_full[idx][tar_index] = cdist(target_pose, measurement_pose, metric='correlation')
                    elif self.similarity_distance == 'jaccard':
                        mapped_meas_full[idx][tar_index] = cdist(target_pose, measurement_pose, metric='jaccard')
                    else:
                        mapped_meas_full[idx][tar_index] = self.get_image_dist(self.similarity_distance, target, measurement, image_path)

                    temp_array.append(mapped_meas_full[idx][tar_index])

            valid_indices = []
            for idx, item in enumerate(measurement_locations):
                if item != -1:
                    valid_indices.append(idx)

            # Convert to likelihood
            mapped_meas_full[valid_indices, tar_index] = softmax(mapped_meas_full[valid_indices, tar_index],axis=0)

            # Reverse to 1-x for similarity
            mapped_meas_full[valid_indices, tar_index] = 1-mapped_meas_full[valid_indices, tar_index]

            # Rescale to [-1,1]
            mapped_meas_full[valid_indices, tar_index]= minmax_scale(mapped_meas_full[valid_indices, tar_index], feature_range=(-1, 1), axis=0, copy=False)

        shape_params = self.map_shape[0:] + [num_targets]

        corr_maps = mapped_meas_full.reshape(shape_params)
        corr_maps = np.pad(corr_maps, ((10, 10), (10, 10), (0, 0)), 'constant', constant_values=0)
        cropped_corr_maps = np.zeros((self.args.crop_size, self.args.crop_size, self.args.max_targets))

        for idx, target_center in enumerate(target_centers):
            t_y = target_center[0][0] + self.args.crop_size // 2
            t_x = target_center[0][1] + self.args.crop_size // 2

            # offset width
            x_top_left = int(t_x - self.args.crop_size // 2 - 1)
            y_top_left = int(t_y - self.args.crop_size // 2 - 1)

            uncropped_map = corr_maps[:,:,idx]
            cropped_map = uncropped_map[y_top_left:y_top_left+self.args.crop_size,x_top_left:x_top_left+self.args.crop_size]

            cropped_corr_maps[:,:,idx] = cropped_map

        # Add batch channel
        cropped_corr_maps = np.expand_dims(cropped_corr_maps, axis = 0)

        return cropped_corr_maps, corr_maps
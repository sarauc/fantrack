import gc
import os
import time
import sys
from fan_track.experiments.ablation import *
import avod.builders.config_builder_util as config_builder
from fan_track.utils.generic_utils import *
import matplotlib.pyplot as plt
import fan_track.network.model as assoc_model
import numpy as np
import fan_track.data_generation.position_shape_utils as ps_utils
import tensorflow as tf
from PIL import Image
from _functools import reduce
from avod.builders.dataset_builder import DatasetBuilder
from avod.core import box_3d_encoder
from avod.core import box_3d_projector
from avod.core.models.avod_model import AvodModel
from avod.core.models.rpn_model import RpnModel
from avod.protos import pipeline_pb2
from google.protobuf import text_format
from fan_track.avod_feature_extractor.bev_cam_feature_extractor import BevCamFeatExtractor
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from fan_track.data_generation.kitti_assocnet_dataset import prepare_data
from fan_track.data_generation.kitti_assocnet_dataset import KittiAssocnetDataset
from fan_track.config.config import TrackerConfig
from fan_track.config.config import GlobalConfig
from scipy.optimize import linear_sum_assignment
from wavedata.tools.core import calib_utils
from fan_track.utils import vis_utils as vis_utils
from fan_track.utils import generic_utils
from sklearn.preprocessing import minmax_scale

class Track:

	def __init__(self, track_id, current_target, imu0_bbox):
		self.track_id = track_id
		self.current_target = current_target
		self.prob_existence = 1.0
		self.age = 1
		self.object_type = 0  # 0 - Car, 1 - pedestrian

		# Kalman Filter Parameters
		self.state_dim = 6
		self.measurement_dim = 3
		self.filter = KalmanFilter(dim_x=self.state_dim, dim_z=self.measurement_dim)

		# Define the Filter State variables (bbox is in imu0 coordinates)
		self.filter.x = np.concatenate((imu0_bbox[0:3], np.zeros_like(current_target.bbox[0:3])))
		# self.filter.x = np.concatenate((imu0_bbox[0:3], np.zeros_like(current_target.bbox[0:3])))

		# Define the State Transition matrix
		self.filter.F = np.eye(self.state_dim, self.state_dim)

		self.filter.F[0:self.measurement_dim, self.measurement_dim:self.state_dim] += np.eye(self.measurement_dim,
																							 self.measurement_dim) * TrackerConfig.DT
		# Define the measurement function
		self.filter.H = np.eye(self.measurement_dim, self.state_dim)

		# Define the measurement covariance
		self.filter.R = self.get_R_matrix()

		# Define the Process covariance
		self.filter.P = self.get_P_matrix()

		# Define the process noise
		self.filter.Q = self.get_Q_matrix()

	def update_matrices(self):
		self.filter.Q = self.get_Q_matrix()
		self.filter.F = np.eye(self.state_dim, self.state_dim)
		self.filter.F[0:self.measurement_dim, self.measurement_dim:self.state_dim] += np.eye(self.measurement_dim,
																							 self.measurement_dim) * TrackerConfig.DT

	def update_state(self, position, velocity):

		# print('\nState at t=0:{0}'.format(self.filter.x))
		# print('\nP Matrix:{0}'.format(str(self.filter.P)))
		self.filter.x = np.concatenate((position, velocity))

	# print('State at t=1:{0}\n'.format(self.filter.x))
	def get_P_matrix(self):

		r_matrix = self.get_R_matrix()

		p_mat = np.zeros(shape=(self.state_dim, self.state_dim), dtype=np.float32)
		p_mat[0:self.measurement_dim, 0:self.measurement_dim] += r_matrix
		p_mat[0:self.measurement_dim, self.measurement_dim:self.state_dim] += (r_matrix / TrackerConfig.DT)
		p_mat[self.measurement_dim:self.state_dim, 0:self.measurement_dim] += (r_matrix / TrackerConfig.DT)
		p_mat[self.measurement_dim:self.state_dim, self.measurement_dim:self.state_dim] += (
					2 * r_matrix / (TrackerConfig.DT) ** 2)

		return p_mat

	def get_R_matrix(self):

		covM = np.zeros(shape=(self.measurement_dim, self.measurement_dim), dtype=np.float32)
		# width and height of the measurement bbox
		covM[0][0] = TrackerConfig.AVOD_X_MSE  # Center X mse
		covM[1][1] = TrackerConfig.AVOD_Y_MSE  # Center Y mse
		covM[2][2] = TrackerConfig.AVOD_Z_MSE  # Center Z mse

		if self.measurement_dim == 6:
			covM[3][3] = TrackerConfig.AVOD_H_MSE
			covM[4][4] = TrackerConfig.AVOD_W_MSE
			covM[5][5] = TrackerConfig.AVOD_L_MSE

		# return elements of the matrix
		return covM[:]

	def get_Q_matrix(self):

		q = TrackerConfig.ACCELERATION_VARIANCE_FACTOR # acceleration component  Should be high for cars
		q_mat = np.zeros(shape=(self.state_dim, self.state_dim), dtype=np.float32)

		q_mat = Q_discrete_white_noise(2, dt=TrackerConfig.DT, var=q, block_size=self.measurement_dim)

		return q_mat

	def update(self, measurement, sim_score, objectness_score, target_score, imu0_bbox):

		self.current_target = measurement

		# Kalman update
		self.filter.update(z=imu0_bbox[0:3])

		kappa = 1 / ((80 - imu0_bbox[4] + 1) * (70 - imu0_bbox[3] + 1) * (8 - imu0_bbox[5] + 1))

		# Update probability of existence
		delta = target_score * (1 - (self.filter.likelihood / (TrackerConfig.AVOD_FAR * kappa)))

		if self.age > 5:
			self.prob_existence = ((1 - delta) / (
				1 - self.prob_existence * delta)) * self.prob_existence
		else:
			self.prob_existence /= TrackerConfig.SURVIVABILITY_FACTOR

	def predict(self):
		self.filter.predict()

class TrackedObject:
	def __init__(self):
		# Tracker Essentials
		self.bbox = np.zeros((6), dtype=np.float32)  # In Camera coordinates
		self.yaw_imu0 = 0.0
		self.camk_bbox = np.zeros((6), dtype=np.float32)
		self.avod_features = np.zeros((7, 7, 640), dtype=np.float32)
		self.reference_frame = 'Camera'
		self.objectness_score = 0.0
		self.track = None
		self.max_velocity = 2.5  # 90 kmph -> 25 m/s -> by ten for ten frames
		self.track_id = -1
		self.object_label = ObjectLabel()
		self.object_label.type = 'DontCare'
		self.occluded = False
		self.truncated = False

	def printToFile(self, file, frame_number):

		file.write(str(frame_number) + ' ' + \
				   str(self.track_id) + ' ' + \
				   self.object_label.type + ' ' + \
				   str(self.object_label.truncation) + ' ' + \
				   str(self.object_label.occlusion) + ' ' + \
				   '%.6f' % self.object_label.alpha + ' ' + \
				   '%.6f' % self.object_label.x1 + ' ' + \
				   '%.6f' % self.object_label.y1 + ' ' + \
				   '%.6f' % self.object_label.x2 + ' ' + \
				   '%.6f' % self.object_label.y2 + ' ' + \
				   '%.6f' % self.object_label.h + ' ' + \
				   '%.6f' % self.object_label.w + ' ' + \
				   '%.6f' % self.object_label.l + ' ' + \
				   '%.6f' % self.object_label.t[0] + ' ' + \
				   '%.6f' % self.object_label.t[1] + ' ' + \
				   '%.6f' % self.object_label.t[2] + ' ' + \
				   '%.6f' % self.object_label.ry + ' ' + \
				   '%.6f' % self.object_label.score + '\n'
				   )

	def update_object_label(self, box_3d, img_path, calib_info):

		# Convert bbox to image coordinates

		image = Image.open(img_path)
		img_box = box_3d_projector.project_to_image_space(box_3d, calib_info,
														  truncate=True, discard_before_truncation=False,
														  image_size=image.size)
		if img_box is not None:
			self.object_label.x1 = img_box[0]
			self.object_label.y1 = img_box[1]
			self.object_label.x2 = img_box[2]
			self.object_label.y2 = img_box[3]
			img_box_updated = True
		else:
			img_box_updated = False

		self.object_label.l = box_3d[3]
		self.object_label.w = box_3d[4]
		self.object_label.h = box_3d[5]

		self.object_label.t = (box_3d[0], box_3d[1], box_3d[2])
		self.object_label.ry = box_3d[6]

		return img_box_updated

	def is_object_inside_image_space(self):

		if (self.object_label.x1 == 0 and
				self.object_label.y1 == 0 and
				self.object_label.x2 == 0 and
				self.object_label.y2 == 0):
			return False
		else:
			return True


def build_global_maps(measurement_locations, m_bbox_feat, m_appear_feat, m_bbox_weights, m_appear_weights,
						t_bbox_feat, t_appear_feat, t_bbox_weights, t_appear_weights, target_centers, num_targets):

	map_size = (GlobalConfig.MAP_LENGTH, GlobalConfig.MAP_WIDTH)
	map_shape = list(map_size)
	serialized_map_shape = [reduce((lambda h, w: h*w), map_size)]

	m_bbox_map = np.zeros(
		serialized_map_shape + [m_bbox_feat.shape[1]])
	m_appear_map = np.zeros(
		serialized_map_shape + [m_appear_feat.shape[1]])

	bbox_weight_map = np.zeros(serialized_map_shape + [1])
	appear_weight_map = np.zeros(serialized_map_shape + [1])

	valid_meas_locations = np.reshape(
		np.argwhere(measurement_locations != -1), [-1])

	for valid_meas_location in (valid_meas_locations):
		measurement_index = measurement_locations[valid_meas_location]

		# index is the 1 dimensional  spatial index in the map
		# val is a measurement index of the features
		m_bbox_map[valid_meas_location, :] = m_bbox_feat[
												measurement_index, :]
		m_appear_map[valid_meas_location, :] = m_appear_feat[
													measurement_index, :]
		bbox_weight_map[valid_meas_location, :] = m_bbox_weights[
													measurement_index, :]
		appear_weight_map[valid_meas_location, :] = m_appear_weights[
													measurement_index, :]

	# Batch dimension - Not really useful but just to preserve the format
	m_bbox_map = np.expand_dims(m_bbox_map, axis=0)
	m_appear_map = np.expand_dims(m_appear_map, axis=0)
	t_bbox_feat = np.expand_dims(t_bbox_feat, axis=0)
	t_appear_feat = np.expand_dims(t_appear_feat, axis=0)

	# Target dimension
	m_bbox_map = np.expand_dims(m_bbox_map, axis=-1)
	m_appear_map = np.expand_dims(m_appear_map, axis=-1)

	corr_bbox_map = np.zeros(
		(m_bbox_map.shape[0], m_bbox_map.shape[1], 1, num_targets))
	corr_appear_map = np.zeros(
		(m_appear_map.shape[0], m_appear_map.shape[1], 1, num_targets))

	for spatial in valid_meas_locations:
		for tar_idx in range(num_targets):
			corr_bbox_map[0, spatial, :, tar_idx] = np.dot(
				t_bbox_feat[0, tar_idx, :], m_bbox_map[0, spatial])
			corr_appear_map[0, spatial, :, tar_idx] = np.dot(
				t_appear_feat[0, tar_idx, :], m_appear_map[0, spatial])

	corr_bbox_map = np.clip(corr_bbox_map, -1.0, 1.0)
	corr_appear_map = np.clip(corr_appear_map, -1.0, 1.0)

	shape_params = map_shape[0:] + [num_targets]
	corr_bbox_map = np.reshape(corr_bbox_map, shape_params)
	corr_appear_map = np.reshape(corr_appear_map, shape_params)

	bbox_weight_map = np.reshape(bbox_weight_map, map_shape[0:])
	appear_weight_map = np.reshape(appear_weight_map, map_shape[0:])

	bbox_weight_map = np.expand_dims(bbox_weight_map, axis=2)
	appear_weight_map = np.expand_dims(appear_weight_map, axis=2)

	bbox_weight_map = np.tile(bbox_weight_map, (1, 1, num_targets))
	appear_weight_map = np.tile(appear_weight_map, (1, 1, num_targets))

	target_bbox_weights_transpose = np.transpose(t_bbox_weights)
	target_appear_weights_transpose = np.transpose(t_appear_weights)

	product_bbox_weight_map = target_bbox_weights_transpose * \
								bbox_weight_map
	product_appear_weight_map = target_appear_weights_transpose * \
								appear_weight_map

	product_bbox_weight_map = product_bbox_weight_map / (
			product_bbox_weight_map + product_appear_weight_map + 1e-10)
	product_appear_weight_map = product_appear_weight_map / (
			product_bbox_weight_map + product_appear_weight_map + 1e-10)

	corr_bbox_map = corr_bbox_map * product_bbox_weight_map
	corr_appear_map = corr_appear_map * product_appear_weight_map

	corr_map = corr_bbox_map + corr_appear_map

	new_height = int(map_shape[0] + (GlobalConfig.CROP_SIZE - 1))
	new_width = int(map_shape[1] + (GlobalConfig.CROP_SIZE - 1))

	pad_width = int((GlobalConfig.CROP_SIZE - 1) / 2)
	n_minus_t = GlobalConfig.MAX_TARGETS - num_targets
	corr_map = np.pad(
		corr_map, pad_width=(
			(pad_width, pad_width), (pad_width, pad_width), (0, 0)),
		mode='constant', constant_values=0)

	crop_w = GlobalConfig.CROP_SIZE
	crop_h = GlobalConfig.CROP_SIZE

	t_x = target_centers[:, :, 0] + crop_w / 2
	t_y = target_centers[:, :, 1] + crop_h / 2

	# offset widths and heights
	x_top_left = t_x - crop_w / 2 - 1
	y_top_left = t_y - crop_h / 2 - 1

	local_corr_map = np.zeros((crop_h, crop_w, GlobalConfig.MAX_TARGETS))

	for tar_idx in range(num_targets):
		x_start = int(x_top_left[tar_idx][0])
		y_start = int(y_top_left[tar_idx][0])
		local_corr_map[:, :, tar_idx] = corr_map[
										x_start:x_start + crop_h, y_start:y_start + crop_w, tar_idx]

	# Dummy batch dimension
	local_corr_map = np.expand_dims(local_corr_map, axis=0)
	return local_corr_map


class Tracker:

	def __init__(self, args, video_no, dataset_name):

		self.init_paths(GlobalConfig.KITTI_ROOT)
		self.number_of_tracks = 0
		self.tracks = []
		self.running_track_id = 0
		self.training_split = 0.80
		self.dataset_name = dataset_name

		# print('Initializing AVOD for video {0}'.format(video_no))
		prepare_data(video_no, self.dataset_name)

		# Tensorflow variables
		self._avod_pred_graph = tf.Graph()
		self._avod_feat_graph = tf.Graph()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self._avod_feat_sess = tf.Session(config=config)

		# Initialize AVOD

		avod_root_dir = '/content/fantrack/fan_track/object_detector/'
	  
		avod_checkpoint = os.path.join(avod_root_dir, 'data', \
						  args.avod_checkpoint_name,'checkpoints', args.avod_checkpoint_name + '-' + TrackerConfig.AVOD_CKPT_NUMBER)
		experiment_config_path = os.path.join(avod_root_dir ,'data', args.avod_checkpoint_name, args.avod_checkpoint_name + '.config')

		avod_pipeline_config = pipeline_pb2.NetworkPipelineConfig()
		with open(experiment_config_path, 'r') as f:
			text_format.Merge(f.read(), avod_pipeline_config)

		dataset_config = config_builder.proto_to_obj(avod_pipeline_config.dataset_config)
		dataset_config.data_split = 'test'
		dataset_config.data_split_dir = 'testing'
		dataset_config.has_labels = False
		dataset_config.dataset_dir = GlobalConfig.KITTI_ROOT + "/object"

		# Remove augmentation during evaluation in test mode
		dataset_config.aug_list = []

		# Build the dataset object
		self.dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
														  use_defaults=False)

		print("Video", video_no, 'Samples:',self.dataset.num_samples)

		# Setup the model
		model_name = avod_pipeline_config.model_config.model_name
		tf.reset_default_graph()
		with self._avod_pred_graph.as_default():
			if model_name == 'avod_model':
				self.avod_model = AvodModel(avod_pipeline_config.model_config,
											train_val_test='test',
											dataset=self.dataset)
			elif model_name == 'rpn_model':
				self.avod_model = RpnModel(avod_pipeline_config.model_config,
										   train_val_test='test',
										   dataset=self.dataset)
			else:
				raise ValueError('Invalid model name {}'.format(model_name))

			self._avod_pred_sess = tf.Session(config=config, graph=self._avod_pred_graph)
			self._avod_prediction_dict = self.avod_model.build()

			self._avod_pred_saver = tf.train.Saver()
			# self._avod_pred_saver  = tf.train.import_meta_graph('/home/v6balasu/repos/deep_tracking/object_detector/avod/data/outputs/cars_reduced_size/checkpoints/cars_reduced_size.meta')
			self._avod_pred_saver.restore(self._avod_pred_sess, avod_checkpoint)
		with self._avod_feat_graph.as_default():
			# avod feature extractor
			self._feat_ext = BevCamFeatExtractor(dataset=self.dataset, model_config=avod_pipeline_config.model_config)

			# prepare the default inputs to the detector
			self._feat_ext.prepare_inputs()

			# construct the nets
			self._feat_ext.setup_bev_img_vggs()
			# self._feat_ext.setup_early_fusion()
			self._feat_ext.setup_expand_predictions()
		args.training_type = TrainingType.SimnetRun
		self.assoc_model_simnet = assoc_model.AssocModel(args)
		self.assoc_model_simnet.build_simnet()
		self.sim_summary_writer = tf.summary.FileWriter(os.path.join(generic_utils.get_project_root(),'data','tensorboard/tracker_runtime/'), self.assoc_model_simnet.graph)

		args.training_type = TrainingType.SimnetRun
		self.assoc_model_assocnet = assoc_model.AssocModel(args)
		self.assoc_model_assocnet.build_assoc_model()
		self.mapper = KittiAssocnetDataset(dataset_name=self.dataset_name)
		self.visualize = TrackerConfig.VISUALIZE
		self.use_gt = TrackerConfig.USE_GT
		self.filter_predictions = []

		# Ablation
		# self.position_based_ablation = PositionBased(self.assoc_model_assocnet.args)

		self.sim_values = []
		# self.debug_file = open('debug.txt','w')

	def init_paths(self, path):
		self.kitti_path = path

	def begin_tracking(self, video_no, dataset_subtype='val'):

		self.video_no = video_no

		inf_path = os.path.join('../data/predictions', self.dataset_name)
		if not os.path.exists(inf_path):
			os.makedirs(inf_path)

		# Prepare Inference results file
		# inf_file = open('../evaluation/test/' + video_no + '.txt', 'w')
		inf_file = open(os.path.join(inf_path, video_no + '.txt'), 'w')

		self.last_known_good_framenumber = 0
		available_transformations = len(self.mapper.oxts_projections[video_no])

		if self.use_gt:
			self.gt_detections = self.get_gt_detections()

		img_path = os.path.join(GlobalConfig.KITTI_ROOT, 'object/testing/image_2/')

		if dataset_subtype == 'val':
			start_frame = int(self.dataset.num_samples * self.training_split) + 1
			end_frame = self.dataset.num_samples - 1
		elif dataset_subtype == 'trainval':
			start_frame = 0
			end_frame = self.dataset.num_samples - 1
		else:
			start_frame = 0
			end_frame = int(self.dataset.num_samples * self.training_split)

		if TrackerConfig.START_FRAME is not None:
			start_frame = TrackerConfig.START_FRAME

		if TrackerConfig.END_FRAME is not None:
			end_frame = TrackerConfig.END_FRAME

		self.first_frame = True
		for frame in range(start_frame, end_frame+1):

			if frame in TrackerConfig.VISUALIZE_RANGE:
				self.visualize = True
			else:
				self.visualize = False

			frame_inference_start = time.time()

			# self.debug_file.write('\nFrame:{0}'.format(frame))

			self.current_frame_name = "%06d" % frame
			print('\n')
			print('Processing Video:{0} Frame:{1}'.format(self.video_no,self.current_frame_name))
			self.frame_number = int(self.current_frame_name)

			# Initialize visualization for the frame
			if self.visualize:
				self.figure, self.ax1, self.ax2 = vis_utils.visualization(img_path, self.frame_number, False, False)

			if self.frame_number < available_transformations:
				self.last_known_good_framenumber = self.frame_number

			# Get Calibration info for the frame
			self.frame_calibration_info = calib_utils.read_calibration(self.dataset.calib_dir, self.frame_number).p2

			# Predict track Existence
			for track in self.tracks:
				# self.debug_file.write('\nTrack_:{0} bb0x:{1}'.format(track.track_id, track.current_target.bbox[0:3]))
				track.prob_existence *= TrackerConfig.SURVIVABILITY_FACTOR

			# Check if it is the first frame
			if self.first_frame:
				measurements = self.get_measurements(video_no, self.frame_number)

				# Init Tracks
				for i, measurement in enumerate(measurements):
					# Assign a new track_id based on 0
					self.create_new_track(measurement)

			else:
				targets = measurements # Now targets are in k-1 th coordinates

				# Now measurements are in k th coordinates
				measurements = self.get_measurements(video_no, self.frame_number)

				# Kalman filter prediction
				self.predict_tracks()

				# After predict_tracks targets are updated to predicted locations
				# i.e. possible k th coordinate locations
				# DataAssociation Task
				associations = self.get_associations(targets, measurements)

				self.update_tracks(targets, measurements, associations)

			for track in self.tracks:

				# Update Track Age
				track.age += 1

				if track.current_target.truncated:
					continue

				# print('Prob exist for track:{0:d} is {1:f}'.format(track.track_id,track.prob_existence))
				track.current_target.printToFile(inf_file, self.frame_number - start_frame)

				# Draw bboxes
				if self.visualize:
					obj = track.current_target.object_label
					vis_utils.draw_box_3d(self.ax2, obj, self.frame_calibration_info,
										  track.current_target.objectness_score, show_orientation=False,
										  track_id=track.track_id)
			# Visualize
			if self.visualize:
				self.visualize_tracking(video_no)

			self.first_frame = False

			inference_time = time.time() - frame_inference_start
			print('Inference time (s):', inference_time)
		self.tracks.clear()
		inf_file.close()

		# self.debug_file.close()

	def create_new_track(self, measurement):

		imu0_bbox = np.copy(measurement.bbox)
		self.mapper.transform_cam_to_imu(self.video_no, self.frame_number, imu0_bbox, is_target=False, toimu0=True)

		track = Track(self.running_track_id, measurement, imu0_bbox)
		self.tracks.append(track)
		measurement.track_id = track.track_id
		measurement.track = track
		self.number_of_tracks += 1
		self.running_track_id += 1

	def destroy_track(self, track):

		if track in self.tracks:
			self.tracks.remove(track)
			self.number_of_tracks -= 1

	def predict_tracks(self):

		for track in self.tracks:

			# Predict track with Kalman Filter
			track.predict()

			# Update the bbox in camera coordinates
			camk_bbox = np.concatenate((track.filter.x[0:3],[1,1,1]))
			camk_bbox = np.append(camk_bbox, track.current_target.yaw_imu0)

			self.mapper.transform_imu0_to_cam(self.video_no, self.frame_number, camk_bbox)

			# Update only the centers. Last three elements are junk
			track.current_target.bbox[0:3] = camk_bbox[0:3]

			# Check if center of target is behind the ego car's camera
			if camk_bbox[2] < 0:
				track.current_target.truncated = True

			# Update the object label in camera coordinates
			is_img_box_updated = track.current_target.update_object_label(track.current_target.bbox,
																		  img_path=self.dataset.get_rgb_image_path(
																			  self.current_frame_name),
																		  calib_info=self.frame_calibration_info)

			if not is_img_box_updated:
				track.current_target.truncated = True

			# Visualize prediction
			if TrackerConfig.VISUALIZE_PREDICTIONS:
				obj = track.current_target.object_label
				if self.visualize:
					vis_utils.draw_box_3d(self.ax2, obj, self.frame_calibration_info, track.current_target.objectness_score,
										track_id=track.track_id, box_color='b', is_prediction=True)

				self.add_to_filter_visualizer(camk_bbox[0], camk_bbox[2], track.track_id)

	def get_associated_targets(self, associations):

		associated_targets = []
		for association in associations:
			if association[1] != None:
				associated_targets.append(association[0])
		return associated_targets

	def update_tracks(self, targets, measurements, associations):

		if associations is not None:

			for item in associations:

				# target available
				if item[0] is not None:
					target_track_id = item[0].track_id

					measurement = item[1]

					# Measurement available
					if measurement is not None:
						measurement.track_id = target_track_id


						# Get the imu0 bbox
						meas_imu_bbox = np.copy(measurement.bbox)
						tar_imu_bbox = np.copy(item[0].bbox)


						self.mapper.transform_cam_to_imu(self.video_no, self.frame_number, meas_imu_bbox, is_target=False,
														 toimu0=True)

						self.mapper.transform_cam_to_imu(self.video_no, self.frame_number-1, tar_imu_bbox, is_target=False,
														 toimu0=True)
						# Update if the track is available
						# Also updates the existence probability of the track
						track_obj = item[0].track
						if track_obj is not None:

							# Update target->track links
							item[0].track = None
							measurement.track = track_obj

							if track_obj.age == 2:
								# Compute Velocities
								velocities = (meas_imu_bbox[0:3] - tar_imu_bbox[0:3]) / TrackerConfig.DT

								track_obj.update_state(meas_imu_bbox[0:3], velocities)

							track_obj.update(measurement, sim_score=item[2],
											 objectness_score=item[0].objectness_score,
											 target_score=item[0].object_label.score, imu0_bbox=meas_imu_bbox)
						else:
							# Target's track was deleted - So create a new track for measurement
							self.create_new_track(item[1])

					# Measurement not available
					else:
						# print('Measurmenet Occluded')
						occluded_target = item[0]
						occluded_target.occluded = True
						occluded_track = occluded_target.track
						if occluded_track is not None and len(measurements) < 21 \
								and occluded_target not in measurements:
							measurements.append(occluded_target)

				# target not available
				else:
					# Simply create a new track
					self.create_new_track(item[1])

		for track in self.tracks:
			if (track.prob_existence < TrackerConfig.EXISTENCE_THRESHOLD or
				not track.current_target.is_object_inside_image_space()) or track.current_target.truncated:

				if track.current_target in measurements:
					measurements.remove(track.current_target)

				track.current_target.track = None
				self.destroy_track(track)

	def get_measurements(self, video_no, frame_number):

		# retrieve inputs to extract targets' features
		self._feat_ext.get_sample(frame_number)

		# Obtain AVOD predictions
		predictions = self.get_predictions_from_avod(frame_number)

		bboxes = predictions['avod_top_prediction_boxes_3d']

		if not self.use_gt:
			predicted_box_corners_and_scores = self.get_avod_predicted_boxes_3d_and_scores(predictions,
																						   box_rep='box_3d')
		else:
			# Ground Truth
			if self.frame_number not in self.gt_detections:
				return []
			else:
				predicted_box_corners_and_scores = self.gt_detections[self.frame_number]

		predicted_box_corners_and_scores = predicted_box_corners_and_scores[
			np.where(predicted_box_corners_and_scores[:, 7] > TrackerConfig.DETECTOR_THRESHOLD)]

		# Consider only the top 21 measurements for association
		predicted_box_corners_and_scores = predicted_box_corners_and_scores[0:20]

		bboxes = predicted_box_corners_and_scores[:, 0:7]

		scores = predicted_box_corners_and_scores[:, 7]
		final_pred_types = predicted_box_corners_and_scores[:, 8]
		objectness_scores = predicted_box_corners_and_scores[:, 9]

		with self._avod_feat_graph.as_default():

			# prepare anchors for the avod
			self._feat_ext.get_anchors(bboxes)

			# extract bev and cam features
			avod_features = self._feat_ext.extract_features()

		measurements = list()
		frame_scoped_index = 0

		# print('Measurements:'+str(len(predicted_box_corners_and_scores)))

		for frame_scoped_index in range(len(predicted_box_corners_and_scores)):
			measurement = TrackedObject()
			pred_type = final_pred_types[frame_scoped_index]
			measurement.objectness_score = objectness_scores[frame_scoped_index]
			measurement.object_label = self.get_object_attributes(bboxes[frame_scoped_index], pred_type,
																  scores[frame_scoped_index])
			measurement.bbox = bboxes[frame_scoped_index]
			measurement.yaw_imu0 = bboxes[frame_scoped_index][6] - self.mapper.delta_yaws[self.video_no][1]
			measurement.bbox = np.asarray(measurement.bbox)
			measurement.avod_features = avod_features[frame_scoped_index]
			measurements.append(measurement)

			# self.debug_file.write('\nm_:{0} {1}'.format(frame_scoped_index,measurement.bbox[0:3]))

			# Visualize detections
			if self.visualize:
				vis_utils.draw_box_3d(self.ax1, measurement.object_label, self.frame_calibration_info,
									  measurement.objectness_score,
									  show_orientation=False)

		return measurements

	def get_gt_detections(self):

		label_path = os.path.join(self.kitti_path,'tracking_dataset', 'labels', 'label_02', self.video_no + '.txt')

		detections = {}

		with open(label_path, 'r') as f:

			frame_id = None
			# read each line of label file
			for line_no, line in enumerate(f):
				# split the string on whitespace to obtain a list of columns
				obj_info = line.split()

				frame_id = int(obj_info[0])

				if (obj_info[2] in ['Car', 'Van']):
					# get the bbox params in the format [h,w,l,x,y,z,ry]
					bbox_ry = np.asarray([obj_info[10:17]], dtype=np.float32)

					# convert Kitti's box to 3d box format [x,y,z,l,w,h,ry]
					bbox_ry = ps_utils.kitti_box_to_box_3d(bbox_ry)[0]

					bbox_ry = np.concatenate((np.asarray(bbox_ry), np.asarray([1, 0, 0.99])))

					if frame_id not in detections:
						detections[frame_id] = np.asarray([bbox_ry])
					else:
						detections[frame_id] = np.append(detections[frame_id], np.asarray([bbox_ry]), axis=0)

		return detections

	def get_object_attributes(self, prediction, pred_type, score):
		object = ObjectLabel()

		pred_type = pred_type.astype(np.int32)
		object.type = self.dataset.classes[pred_type]
		object.alpha = -10
		object.truncation = -1
		object.occlusion = -1

		# Convert bbox to image coordinates
		img_idx = int(self.current_frame_name)
		image = Image.open(self.dataset.get_rgb_image_path(self.current_frame_name))

		stereo_calib_p2 = calib_utils.read_calibration(self.dataset.calib_dir,
													   img_idx).p2

		box_3d = np.copy(prediction[0:7])

		img_box = box_3d_projector.project_to_image_space(box_3d, stereo_calib_p2,
														  truncate=True, discard_before_truncation=False,
														  image_size=image.size)

		obj_label = box_3d_encoder.box_3d_to_object_label(box_3d)

		if img_box is not None:
			object.x1 = img_box[0]
			object.y1 = img_box[1]
			object.x2 = img_box[2]
			object.y2 = img_box[3]

		object.l = prediction[3]
		object.w = prediction[4]
		object.h = prediction[5]

		object.t = (prediction[0], prediction[1], prediction[2])
		object.ry = prediction[6]

		object.score = score

		return object

	def get_predictions_from_avod(self, frame_number):
		with self._avod_pred_graph.as_default():
			feed_dict = self.avod_model.create_feed_dict(sample_index=frame_number)

			sample_name = self.avod_model.sample_info['sample_name']

			predictions = self._avod_pred_sess.run(self._avod_prediction_dict,
												   feed_dict=feed_dict)
		return predictions

	def transform_bb_imu0_to_imuk(self, objects, is_target=False):

		imuk_bboxes = []
		for obj in objects:

			# Convert bboxes to imuk
			imugps_bbox = obj.bbox.copy()

			# If this is measurement transform to k-1 frame
			if not is_target:
				self.mapper.transform_imu0_to_imuk(self.video_no, self.frame_number - 1, bbox=imugps_bbox)

			else:
				self.mapper.transform_imu0_to_imuk(self.video_no, self.frame_number, bbox=imugps_bbox)

			imuk_bboxes.append(imugps_bbox)

		imuk_bboxes = np.asarray(imuk_bboxes)
		imuk_bboxes = np.expand_dims(imuk_bboxes, axis=1)

		return imuk_bboxes

	def transform_bb_cam_to_imuk(self, objects, is_target=False):

		imuk_bboxes = []
		for obj in objects:

			# Convert bboxes to imuk
			imugps_bbox = obj.bbox.copy()

			# If this is measurement transform to k-1 frame
			if not is_target:
				self.mapper.transform_cam_to_imu(self.video_no, self.frame_number - 1, imugps_bbox, is_target=is_target)
			else:
				self.mapper.transform_cam_to_imu(self.video_no, self.frame_number, imugps_bbox, is_target=is_target)

			imuk_bboxes.append(imugps_bbox)

		imuk_bboxes = np.asarray(imuk_bboxes)
		imuk_bboxes = np.expand_dims(imuk_bboxes, axis=1)

		return imuk_bboxes

	def get_associations(self, targets, measurements):

		na_present, associations = self.check_none_associations(targets, measurements)
		if na_present:
			return associations

		target_bboxes = self.transform_bb_cam_to_imuk(targets, is_target=True)

		# TODO: Revert the experiment: Don't transform meas to k-1 coordinates
		measurement_bboxes = self.transform_bb_cam_to_imuk(measurements, is_target=True)

		# Prepare simnet Inputs
		measurement_locations, target_centers, unmapped_measurement_indices, unmapped_target_indices = \
			self.prepare_simnet_inputs(target_bboxes, measurement_bboxes)

		measurements = np.delete(measurements, obj=unmapped_measurement_indices, axis=0)
		measurement_bboxes = np.delete(measurement_bboxes, obj=unmapped_measurement_indices, axis=0)

		# Targets that go out of the map
		for unmapped_index in unmapped_target_indices:
			# Destroy the track
			track = targets[unmapped_index].track
			targets[unmapped_index].track = None
			self.destroy_track(track)

		targets = np.delete(targets, obj=unmapped_target_indices, axis=0)
		target_bboxes = np.delete(target_bboxes, obj=unmapped_target_indices, axis=0)

		# Check if targets and measurements became zero again
		na_present, associations = self.check_none_associations(targets, measurements)
		if na_present:
			return associations

		target_avod_feat = np.asarray([target.avod_features for target in targets])
		measurement_avod_feat = np.asarray([measurement.avod_features for measurement in measurements])
		combined_bboxes = np.concatenate((target_bboxes, measurement_bboxes), axis=0)
		combined_bboxes = np.expand_dims(combined_bboxes, axis=3)
		target_bboxes = np.expand_dims(target_bboxes, axis=3)
		measurement_bboxes = np.expand_dims(measurement_bboxes, axis=3)
		combined_avod_feat = np.concatenate((target_avod_feat, measurement_avod_feat), axis=0)

		if len(target_centers) == 0:
			for measurement in measurements:
				associations.append((None, measurement, 0))
			return associations

		start_simnet_time = time.time()

		# Ablation Study
		# self.position_based_ablation.similarity_distance = 'manhattan'
		# image_path = self.dataset.get_rgb_image_path(self.current_frame_name)
		# local_corr_map, corr_maps = self.position_based_ablation.euclidean_similarity(combined_bboxes,len(targets),measurement_locations, target_centers, targets, measurements, image_path)
		#
		# print('Euclidean non zero values:',np.around(local_corr_map[np.nonzero(local_corr_map)],decimals=2))

		# print('Time for Ablasion:',time.time()-start_time_ablasion)

		# Ablation: Turn off Avod Features:
		# combined_avod_feat = np.zeros_like(combined_avod_feat)

		# combined_avod_feat = np.random.rand(combined_avod_feat.shape[0],
		# 									combined_avod_feat.shape[1],
		# 									combined_avod_feat.shape[2],
		# 									combined_avod_feat.shape[3])

		# combined_avod_feat = np.random.rand()

		# Ablation: Turn off bbox featuers:
		# combined_bboxes = np.zeros_like(combined_bboxes)

		# combined_bboxes = np.random.rand(combined_bboxes.shape[0],
		# 									combined_bboxes.shape[1],
		# 									combined_bboxes.shape[2],
		# 									combined_bboxes.shape[3])

		# Run Simnet
		m_bbox_feat, m_appear_feat, m_bbox_weights, m_appear_weights, \
		t_bbox_feat, t_appear_feat, t_bbox_weights, t_appear_weights = self.assoc_model_simnet.sess.run(
			[self.assoc_model_simnet.m_feat1,
			 self.assoc_model_simnet.m_feat2,
			 self.assoc_model_simnet.w_bbox_m,
			 self.assoc_model_simnet.w_appear_m,
			 self.assoc_model_simnet.t_feat1,
			 self.assoc_model_simnet.t_feat2,
			 self.assoc_model_simnet.w_bbox_t,
			 self.assoc_model_simnet.w_appear_t],
			feed_dict={
				self.assoc_model_simnet.in_bbox_tar_ph: target_bboxes,
				self.assoc_model_simnet.in_bbox_meas_ph: measurement_bboxes,
				self.assoc_model_simnet.in_appear_tar_ph: target_avod_feat,
				self.assoc_model_simnet.in_appear_meas_ph: measurement_avod_feat,
				self.assoc_model_simnet.num_target_ph: [len(targets)],
				self.assoc_model_simnet.training_ph: False})

		second_half_start = time.time()

		local_corr_map = build_global_maps(measurement_locations, m_bbox_feat, m_appear_feat, m_bbox_weights, m_appear_weights,
						   t_bbox_feat, t_appear_feat, t_bbox_weights, t_appear_weights, target_centers,
						   len(targets))

		print('Simnet_time:',time.time() - start_simnet_time)

		start_assoc_time = time.time()
		# Ablation
		# Hungarian method
		# cost_matrix = np.zeros((len(targets),len(self.mapper.m_centers)))
		# measurement_locations_matrix = measurement_locations.reshape((160,160))
		# nonzeros = np.nonzero(measurement_locations_matrix+1)
		# for tar_idx, target_center in enumerate(target_centers):
		#     t_y = target_center[0][0] + GlobalConfig.CROP_SIZE // 2
		#     t_x = target_center[0][1] + GlobalConfig.CROP_SIZE // 2
		#
		#     # offset width
		#     x_top_left = int(t_x - GlobalConfig.CROP_SIZE // 2 - 1)
		#     y_top_left = int(t_y - GlobalConfig.CROP_SIZE // 2 - 1)
		#
		#     target_local_map = local_corr_map[0, :, :, tar_idx]
		#     target_local_map = -target_local_map
		#
		#     for row_idx in range(target_local_map.shape[0]):
		#         for col_idx in range(target_local_map.shape[1]):
		#             if target_local_map[row_idx][col_idx] != 0:
		#
		#                 cost = target_local_map[row_idx][col_idx]
		#                 global_m_row = y_top_left + row_idx - self.assoc_model_assocnet.args.crop_size // 2
		#                 global_m_col = x_top_left + col_idx - self.assoc_model_assocnet.args.crop_size // 2
		#
		#                 meas_idx = measurement_locations_matrix[global_m_row,global_m_col]
		#                 cost_matrix[tar_idx, meas_idx] = cost
		# row_ind, col_ind = linear_sum_assignment(cost_matrix)
		#
		# m_wise_as_dict = {}
		# t_wise_as_dict = {}
		#
		# for idx in range(len(targets)):
		#     t_wise_as_dict[idx] = [idx, None]
		# for idx in range(len(self.mapper.m_centers)):
		#     m_wise_as_dict[idx] = [None, idx]
		#
		# associations = []
		# for i,val in enumerate(row_ind):
		#     associations.append([val,col_ind[i]])
		#     del t_wise_as_dict[val]
		#     del m_wise_as_dict[col_ind[i]]
		#
		# for m_unassoc in list(m_wise_as_dict.values()):
		#     associations.append(m_unassoc)
		#
		# for t_unassoc in list(t_wise_as_dict.values()):
		#     associations.append(t_unassoc)
		#
		# # End Hungarian Ablation

		# Run Assocnet
		with self.assoc_model_assocnet.graph.as_default():

			m_pred_x, m_pred_y, _ = self.assoc_model_assocnet.sess.run(
				[self.assoc_model_assocnet.x_pred,
				 self.assoc_model_assocnet.y_pred, self.assoc_model_assocnet.sliced_maps],
				feed_dict={
					self.assoc_model_assocnet.in_local_corr_map_ph: local_corr_map,
					self.assoc_model_assocnet.in_num_target_array_ph: [len(targets)],
					self.assoc_model_assocnet.in_training_assocnet_ph: False
				}
			)

		associations, corr_scores = self.get_association_results(target_centers,
																	 len(targets),
																	 self.mapper.m_centers,
																	 m_pred_x,
																	 m_pred_y,
																	 local_corr_map)

		print('Assocnet_time:',time.time() - start_assoc_time)

		# remap to objects
		obj_associations = []

		# self.debug_file.write('\nAssociations:')
		for idx, item in enumerate(associations):

			if item[0] is not None and item[1] is not None:
				# self.debug_file.write('(trk:{0},m_{1})'.format(targets[item[0]].track_id,item[1]))
				obj_associations.append((targets[item[0]], measurements[item[1]], corr_scores[idx]))
			elif item[0] is None:
				# self.debug_file.write('({0},{1})'.format(None, item[1]))
				obj_associations.append((None, measurements[item[1]], 0))
			elif item[1] is None:
				# self.debug_file.write('(trk:{0},{1})'.format(targets[item[0]].track_id, None))
				obj_associations.append((targets[item[0]], None, 0))
			# self.debug_file.write(', ')

		# stub
		return obj_associations

	def check_none_associations(self, targets, measurements):
		associations = []

		if len(targets) == 0 and len(measurements) == 0:
			return True, None

		if len(measurements) == 0:
			for target in targets:
				associations.append((target, None, 0))
			return True, associations

		if len(targets) == 0:
			for measurement in measurements:
				associations.append((None, measurement, 0))
			return True, associations

		return False, associations

	def get_association_results(self, t_centers, num_targets, m_centers, m_pred_x, m_pred_y, corr_maps):
		'''
					Obtain the assignment of measurement labels to target labels.
					:Arguments:
						t_centers: array of target indices in the global map.
						num_targets: the number of targets being tracked.
						m_centers: array of measurement indices in the global map.
						corr_maps: local correlation maps for each targets.
						m_pred_x: the row indices of the predicted pixels in local maps.
						m_pred_y: the column indices of the predicted pixels in local maps.
					:Returns:
						A list of 2-tuples showing target and measurement indices like (i,j) where
						i and j are orders of targets and measurements in the inputs to the network.

				'''

		# target to measurement association scores
		assoc_results = []

		# list of correlation scores for associations
		corr_scores = []

		m_wise_as_dict = {}
		assoc_scores = []

		for m_idx, z_xy in enumerate(m_centers):
			m_wise_as_dict[tuple(z_xy)] = [None,m_idx]

		for t_idx in range(num_targets):

			# local row and column indices of the measurement associated with target with the index  t_idx
			z_x = m_pred_x[t_idx]
			z_y = m_pred_y[t_idx]

			# targets are centered at local maps
			tar_pos_idx = int((GlobalConfig.CROP_SIZE - 1) / 2 + 1)

			# compute the global indices of the measurement
			delta_x = z_x - tar_pos_idx
			delta_y = z_y - tar_pos_idx

			z_global_x = t_centers[t_idx][0, 0] + delta_x
			z_global_y = t_centers[t_idx][0, 1] + delta_y

			if (z_global_x, z_global_y) in m_wise_as_dict:
				corr_scores.append(corr_maps[0][z_x, z_y, t_idx])

				# if there isn't already an association with this measurement
				if m_wise_as_dict[(z_global_x, z_global_y)][0] is None:
					m_wise_as_dict[(z_global_x, z_global_y)][0] = t_idx
				else:
					prev_t_idx = m_wise_as_dict[(z_global_x, z_global_y)][0]

					# compare the corr scores of the associations with these two targets
					if corr_scores[prev_t_idx] < corr_scores[-1]:
						m_wise_as_dict[(z_global_x, z_global_y)][0] = t_idx
						assoc_results.append([prev_t_idx, None])
					else:
						assoc_results.append([t_idx, None])
						corr_scores[-1] = None

					assoc_scores.append(None)

			else:
				assoc_results.append([t_idx, None])
				corr_scores.append(None)
				assoc_scores.append(None)

		m_wise_associations = list(m_wise_as_dict.values())

		for assoc in m_wise_associations:
			if assoc[0] is None:
				assoc_scores.append(None)
			else:
				assoc_scores.append(corr_scores[assoc[0]])

			assoc_results.append(assoc)

		return assoc_results, assoc_scores

	def prepare_simnet_inputs(self, target_bboxes, measurement_bboxes):

		measurement_bboxes = np.reshape(measurement_bboxes, (measurement_bboxes.shape[0], measurement_bboxes.shape[2]))
		measurement_bboxes = measurement_bboxes[:, 0:3]
		target_bboxes = np.reshape(target_bboxes, (target_bboxes.shape[0], target_bboxes.shape[2]))
		target_bboxes = target_bboxes[:, 0:3]

		self.mapper.loc_ind_coords(measurement_bboxes)
		unmapped_measurement_indices = [i for i, x in enumerate(self.mapper.mapped_objects) if not x]
		self.mapper.map_targets(target_bboxes)
		target_centers = np.expand_dims(self.mapper.t_centers, axis=1)

		unmapped_target_indices = [i for i, x in enumerate(self.mapper.mapped_objects) if not x]

		return self.mapper.meas_loc, target_centers, unmapped_measurement_indices, unmapped_target_indices

	def visualize_tracking(self, video_no):

		save_path = os.path.join(generic_utils.get_project_root(), 'data/images', self.dataset_name, video_no)

		# Render results
		if not os.path.exists(save_path):
			os.makedirs(save_path)

		# Remove existing files
		if self.first_frame:
			[os.remove(save_path + '/' + f) for f in os.listdir(save_path) if os.path.isfile(save_path + '/' + f)]
		dest_path = os.path.join(save_path, self.current_frame_name + '.png')
		print('Saving to ',dest_path)
		self.figure.savefig(dest_path, format='png', quality=20)

	def get_avod_predicted_boxes_3d_and_scores(self, predictions,
											   box_rep):
		"""Returns the predictions and scores stacked for saving to file.

		Args:
			predictions: A dictionary containing the model outputs.
			box_rep: A string indicating the format of the 3D bounding
				boxes i.e. 'box_3d', 'box_8c' etc.

		Returns:
			predictions_and_scores: A numpy array of shape
				(number_of_predicted_boxes, 9), containing the final prediction
				boxes, orientations, scores, and types.
		"""

		if box_rep == 'box_3d':
			# Convert anchors + orientation to box_3d
			final_pred_anchors = predictions[
				AvodModel.PRED_TOP_PREDICTION_ANCHORS]
			final_pred_orientations = predictions[
				AvodModel.PRED_TOP_ORIENTATIONS]

			final_pred_boxes_3d = box_3d_encoder.anchors_to_box_3d(
				final_pred_anchors, fix_lw=True)
			final_pred_boxes_3d[:, 6] = final_pred_orientations

		elif box_rep in ['box_8c', 'box_8co', 'box_4c']:
			# Predictions are in box_3d format already
			final_pred_boxes_3d = predictions[
				AvodModel.PRED_TOP_PREDICTION_BOXES_3D]

		elif box_rep == 'box_4ca':
			# boxes_3d from boxes_4c
			final_pred_boxes_3d = predictions[
				AvodModel.PRED_TOP_PREDICTION_BOXES_3D]

			# Predicted orientation from layers
			final_pred_orientations = predictions[
				AvodModel.PRED_TOP_ORIENTATIONS]

			# Calculate difference between box_3d and predicted angle
			ang_diff = final_pred_boxes_3d[:, 6] - final_pred_orientations

			# Wrap differences between -pi and pi
			two_pi = 2 * np.pi
			ang_diff[ang_diff < -np.pi] += two_pi
			ang_diff[ang_diff > np.pi] -= two_pi

			def swap_boxes_3d_lw(boxes_3d):
				boxes_3d_lengths = np.copy(boxes_3d[:, 3])
				boxes_3d[:, 3] = boxes_3d[:, 4]
				boxes_3d[:, 4] = boxes_3d_lengths
				return boxes_3d

			pi_0_25 = 0.25 * np.pi
			pi_0_50 = 0.50 * np.pi
			pi_0_75 = 0.75 * np.pi

			# Rotate 90 degrees if difference between pi/4 and 3/4 pi
			rot_pos_90_indices = np.logical_and(pi_0_25 < ang_diff,
												ang_diff < pi_0_75)
			final_pred_boxes_3d[rot_pos_90_indices] = \
				swap_boxes_3d_lw(final_pred_boxes_3d[rot_pos_90_indices])
			final_pred_boxes_3d[rot_pos_90_indices, 6] += pi_0_50

			# Rotate -90 degrees if difference between -pi/4 and -3/4 pi
			rot_neg_90_indices = np.logical_and(-pi_0_25 > ang_diff,
												ang_diff > -pi_0_75)
			final_pred_boxes_3d[rot_neg_90_indices] = \
				swap_boxes_3d_lw(final_pred_boxes_3d[rot_neg_90_indices])
			final_pred_boxes_3d[rot_neg_90_indices, 6] -= pi_0_50

			# Flip angles if abs difference if greater than or equal to 135
			# degrees
			swap_indices = np.abs(ang_diff) >= pi_0_75
			final_pred_boxes_3d[swap_indices, 6] += np.pi

			# Wrap to -pi, pi
			above_pi_indices = final_pred_boxes_3d[:, 6] > np.pi
			final_pred_boxes_3d[above_pi_indices, 6] -= two_pi

		else:
			raise NotImplementedError('Parse predictions not implemented for',
									  box_rep)

		# Append score and class index (object type)
		final_pred_softmax = predictions[
			AvodModel.PRED_TOP_CLASSIFICATION_SOFTMAX]

		# Find max class score index
		not_bkg_scores = final_pred_softmax[:, 1:]
		final_pred_types = np.argmax(not_bkg_scores, axis=1)

		# Take max class score (ignoring background)
		final_pred_scores = np.array([])
		for pred_idx in range(len(final_pred_boxes_3d)):
			all_class_scores = not_bkg_scores[pred_idx]
			max_class_score = all_class_scores[final_pred_types[pred_idx]]
			final_pred_scores = np.append(final_pred_scores, max_class_score)

		objectness_scores = predictions[RpnModel.PRED_TOP_OBJECTNESS_SOFTMAX][0:len(final_pred_scores)]

		# Stack into prediction format
		predictions_and_scores = np.column_stack(
			[final_pred_boxes_3d,
			 final_pred_scores,
			 final_pred_types, objectness_scores])

		return predictions_and_scores

	def add_to_filter_visualizer(self, x, z, trackid):
		plt.scatter(x, z, cmap=trackid)

	def write_to_video(self, video_no):

		# Define the codec and create VideoWriter object
		fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
		video_file = self.kitti_path + 'results/' + self.dataset_name + '/' + video_no
		self.video_writer = cv2.VideoWriter(video_file + '.avi', fourcc, 5, (1500, 915))

		result_path = self.kitti_path + 'results/' + self.dataset_name + '/' + video_no + '/'
		_, _, files = next(os.walk(result_path))

		for file in files:
			img = cv2.imread(os.path.join(result_path, file))
			cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			self.video_writer.write(img)

		cv2.destroyAllWindows()

		self.video_writer.release()

def visualize_map(local_corr_map):

	for t_idx in range(local_corr_map.shape[3]):

		map = local_corr_map[0,:,:,t_idx]
		map[10,10] = 1
		plt.imshow(map,origin = 'lower')
		plt.show()

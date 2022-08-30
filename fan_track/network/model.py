import tensorflow as tf
from tensorflow.python import debug as tf_debug
from _functools import reduce
import tensorflow.contrib.slim as slim
from collections import OrderedDict
from fan_track.network.layers import max_indices, correct_predictions
from fan_track.utils.generic_utils import TrainingType
from fan_track.config.config import *
import numpy as np
import os

class AssocModel(object):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 44
    config.inter_op_parallelism_threads = 44# High performance

    def __init__(self, args):
        """
        Model for data association in tracking
        """

        self.summaries = True
        self.args = args
        self.channels = self.args.max_targets

        self.graph = tf.Graph()
        map_size = (args.map_length, args.map_width)

        # the shape of the 3d bbox feature vectors
        self.pos_shp_shape = [None, 1, 7, 1]

        # the shape of the avod appearance feature matrices
        self.appear_shape = [None, 7, 7, 640]

        # the shape of the label vector
        self.lab_shape = [None,1]

        # the shape of the target x-y centers
        self.cntr_xy_shape = [None, 1, 2]

        # the shape of the correlation map
        self.corr_map_shape = [1, GlobalConfig.CROP_SIZE, GlobalConfig.CROP_SIZE, self.channels]

        # the shape of the label maps
        self.label_shape = [1,  GlobalConfig.CROP_SIZE, GlobalConfig.CROP_SIZE, self.channels]

        # the shape of the measurement location indicator
        self.loc_shape = [reduce((lambda h, w: h*w), map_size)]

        # the shape of the target map
        self.map_shape = list(map_size)
        self.train_epoches = args.epochs
        self.display_rate = args.disp_rate
        self.ckpt_path = args.save_path
        self.save_rate = args.save_rate
        self.training_type = args.training_type

        # maximum number of target
        self.max_target = args.max_targets

        # weights of negatives in the loss because of skewness
        self.skewness = args.simnet_skewness

        self.sess = tf.Session(config=AssocModel.config, graph=self.graph)
        # self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, 'localhost:6064')

    def save_model(self):
        writer = tf.summary.FileWriter(logdir=self.ckpt_path+'/combined', graph = self.graph)
        writer.flush()

    def build_simnet(self):

        if self.training_type == TrainingType.Inference:
            with self.graph.as_default():
                    saver = tf.train.import_meta_graph(SimnetConfig.SIMNET_CKPT_PATH + '.meta')
                    saver.restore(self.sess, SimnetConfig.SIMNET_CKPT_PATH)
        else:
            with self.graph.as_default():

                # Simnet Variables
                with tf.variable_scope('input'):
                    self.in_bbox_tar_ph = tf.placeholder(tf.float32, self.pos_shp_shape, 'in_bbox_tar')
                    self.in_bbox_meas_ph = tf.placeholder(tf.float32, self.pos_shp_shape, 'in_bbox_meas')
                    self.in_appear_tar_ph = tf.placeholder(tf.float32, self.appear_shape, 'in_appear_tar')
                    self.in_appear_meas_ph = tf.placeholder(tf.float32, self.appear_shape, 'in_appear_meas')
                    self.simnet_labels_ph = tf.placeholder(tf.float32, self.lab_shape, 'labels')
                    self.training_ph = tf.placeholder(tf.bool, name = 'training')
                    self.num_target_ph = tf.placeholder(tf.int32,shape=[1], name = 'num_targets')
                    self.global_step = tf.placeholder(tf.int32, name = 'global_step')

                    # if self.training_type in [TrainingType.Simnet, TrainingType.SimnetRun]:
                    #     self.in_meas_loc_ph = tf.placeholder(tf.int32, self.loc_shape, 'meas_loc_indicator')
                    #     self.in_tar_cntr_xy = tf.placeholder(tf.float32, self.cntr_xy_shape, 'target_center_xy')

                    self.in_bbox_ph = tf.concat([self.in_bbox_tar_ph,self.in_bbox_meas_ph], axis=0)
                    self.in_appear_ph = tf.concat([self.in_appear_tar_ph,self.in_appear_meas_ph], axis=0)
                    self.num_targets = self.num_target_ph[0]

                with slim.arg_scope( self.simnet_arg_scope(mode = self.training_ph) ):

                    '''bbox branch'''

                    # cross channel pooling of 3d bbox params
                    self.conv1_0_act = slim.conv2d(self.in_bbox_ph, 256, 1, scope = 'conv1_0')

                    # flattens the bbox maps into 1D for each batch
                    net1 = slim.flatten(self.conv1_0_act,  scope = 'feat_1d_1')

                    # fc layers with dropout for bbox features
                    net1 = slim.dropout(net1, keep_prob=0.5, scope = 'dropout1')
                    self.fc1_act = slim.fully_connected(net1, 512, scope='fc1')

                    net1 = slim.dropout(self.fc1_act, keep_prob=0.5, scope = 'dropout2')
                    self.fc2_act = slim.fully_connected(net1, 512,  scope='fc2')

                    # remove the feature distribution change before normalization
                    features1 = slim.batch_norm(self.fc2_act,scope='ball1')

                    # project thself.training_phe features to unit hypersphere
                    self.unit_features1 = tf.nn.l2_normalize(features1, axis=1)

                    '''appearance branch'''
                    # fc layer implemented by a conv layer for avod's features
                    self.conv2_0_act = slim.conv2d(self.in_appear_ph, 256, 3, scope = 'conv2_0')

                    ''' To do: The fc layer is implemented using 7x7 kernel and valid padding'''

                    # Global Average Pooling
                    # self.conv2_1_act = tf.layers.average_pooling2d(inputs=self.conv2_0_act, pool_size=7, strides=7)
                    self.conv2_1_act = tf.nn.avg_pool(self.conv2_0_act,ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')

                    # fc layer implemented by a conv layer for avod's features
                    # self.conv2_1_act = slim.conv2d(self.conv2_0_act, 256, 1, scope = 'conv2_1')

                    # flattens the bbox maps into 1D for each batch
                    net2 = slim.flatten(self.conv2_1_act,  scope = 'feat_1d_2')

                    # fc layers with dropout for bbox features
                    net2 = slim.dropout(net2, keep_prob=0.5, scope = 'dropout3')
                    self.fc3_act = slim.fully_connected(net2, 512, scope='fc3')

                    net2 = slim.dropout(self.fc3_act, keep_prob=0.5, scope = 'dropout4')
                    self.fc4_act = slim.fully_connected(net2, 512, scope='fc4')

                    # remove the feature distribution change before normalization
                    features2 = slim.batch_norm(self.fc4_act,scope='ball2')

                    # project the features to unit hypersphere
                    self.unit_features2 = tf.nn.l2_normalize(features2, axis=1)

                # slice bbox feature tensor into targets and measurements
                self.t_feat1 = tf.slice(self.unit_features1,[0,0],[self.num_targets,-1], 't_feat1')

                self.m_feat1 = tf.slice(self.unit_features1,[self.num_targets,0],[-1,-1], 'm_feat1')

                # slice appearance feature tensor into targets and measurements
                self.t_feat2 = tf.slice(self.unit_features2,[0,0],[self.num_targets,-1], 't_feat2')

                self.m_feat2 = tf.slice(self.unit_features2,[self.num_targets,0],[-1,-1], 'm_feat2')

                # compute the weights of bbox and appearance similarities
                concat_feat = tf.concat((self.unit_features1,self.unit_features2), axis=1)

                logits = slim.fully_connected(concat_feat, 2, scope='logits')

                branch_weights = tf.nn.softmax(logits, axis=1, name='branch_weights')

                # slice normalized weights into bbox and appearance weights for targets and measurements
                self.w_bbox_t = tf.slice(branch_weights,[0,0],[self.num_targets,1], 'w_bbox_t')

                self.w_appear_t = tf.slice(branch_weights,[0,1],[self.num_targets,1], 'w_appear_t')

                self.w_bbox_m = tf.slice(branch_weights,[self.num_targets,0],[-1,1], 'w_bbox_m')

                self.w_appear_m = tf.slice(branch_weights,[self.num_targets,1],[-1,1], 'w_appear_m')

                #  create the similarity maps for each target
                    # self.create_sim_map()

                # define the loss, optimizer and predictor
                if self.training_type == TrainingType.Simnet or self.training_type == TrainingType.Both:
                    self.build_simnet_loss_function()

                self.simnet_saver = tf.train.Saver(max_to_keep = self.args.max_to_keep)

                # restore the weights from the checkpoint
                if self.training_type in (TrainingType.Assocnet, TrainingType.SimnetRun, TrainingType.Inference):
                    self.simnet_saver.restore(self.sess, self.args.simnet_ckpt_path)

    def build_simnet_loss_function(self):
        '''
            Define the loss function by measuring how far off the network
            predictions are from the correct answers, i.e., labels. Ask
            the tensorflow to use the Adam optimizer in order to reduce
            the loss and also define the ops to evaluate the accuracy of
            the network.
        '''

        with tf.variable_scope('loss_opt_pred'):

            '''similarity'''
            # compute the cosine similarities
            self.cosine_sim1 = tf.reduce_sum( tf.multiply(self.t_feat1, self.m_feat1), 1, keepdims=True)

            self.cosine_sim2 = tf.reduce_sum( tf.multiply(self.t_feat2, self.m_feat2), 1, keepdims=True)

            # compute weights for cosine similarities
            self.w_bbox = tf.multiply(self.w_bbox_t, self.w_bbox_m, 'weight_bbox')

            self.w_appear = tf.multiply(self.w_appear_t, self.w_appear_m, 'weight_appear')

            # normalize weights for each target-measurement pair
            w_norm_const = self.w_bbox + self.w_appear

            self.w_bbox = self.w_bbox / w_norm_const

            self.w_appear = self.w_appear / w_norm_const

            # decision fusion
            self.cosine_sim =  tf.add(self.w_bbox*self.cosine_sim1,self.w_appear*self.cosine_sim2,'cosine_sim')

            # clipcosine_sim to a value in [-1,1]
            self.cosine_sim = tf.clip_by_value(self.cosine_sim, -1.0, 1.0)

            # model evaluation: accuracy metric
            pred_pos = tf.cast(self.cosine_sim > SimnetConfig.ACCURACY_THRESHOLD, tf.int32, 'pred_positives')

            pred_neg = tf.cast(self.cosine_sim  <= -1*SimnetConfig.ACCURACY_THRESHOLD, tf.int32, 'pred_negatives')

            predictions = tf.add(pred_pos, -1*pred_neg, 'predictions')

            # element-wise comparison to find real trues
            self.comparison = tf.equal(predictions, tf.to_int32(self.simnet_labels_ph), 'comparison')

            # mean of the number of correct predictions
            self.simnet_accuracy = tf.reduce_mean(tf.to_float(self.comparison), name = 'accuracy')

            def f(x):

                y = tf.cond(tf.greater_equal(x,0), lambda : x, lambda: tf.add(x,1))

                return y

            # compute labels for precision and rec
            labels = tf.map_fn(f, tf.reshape(self.simnet_labels_ph, shape=[-1]), dtype=tf.float32)

            '''precision and recall'''
            # map predictions to {0,1}
            norm_predictions =  0.5*(tf.add(tf.to_float(tf.reshape(predictions,shape=[-1])),
                                                        tf.ones(shape=[self.num_targets])
                                            )
                                    )

            self.TP = tf.count_nonzero(norm_predictions * labels, name='true_positives')
            self.TN = tf.count_nonzero((norm_predictions -1) * (labels - 1), name='true_negatives')
            self.FP = tf.count_nonzero(norm_predictions * (labels - 1), name='false_positives')
            self.FN = tf.count_nonzero((norm_predictions - 1) * labels, name='false_negatives')

            self.simnet_precision = tf.div(self.TP, (self.TP + self.FP), name='simnet_precision')
            self.simnet_recall = tf.div(self.TP, (self.TP + self.FN), name='simnet_recall')

            '''loss and optimizer'''
            # considering the skewness between positives and negatives weight contributions to the loss
            angular_dist = 1 - tf.acos(self.cosine_sim, 'angular_similarities')/np.pi

            zeros = tf.zeros_like(self.simnet_labels_ph)

            ones = tf.ones_like(self.simnet_labels_ph)

            pos_examples_indices = tf.where(self.simnet_labels_ph > zeros, ones, zeros)

            neg_examples_indices =  1 - pos_examples_indices

            # we add a small constant 1e-10 to prevent taking of log(0)
            importance = -tf.log(angular_dist + 1e-10)*pos_examples_indices + \
                          -tf.log(1 - angular_dist + 1e-10)*neg_examples_indices

            # the constants in weight computations are obtained using skewnness ratio 18:25
            pos_weights = (18/43)*pos_examples_indices

            neg_weights = (25/43)*neg_examples_indices

            self.weights = tf.multiply(pos_weights + neg_weights, importance)

            # compute the mean of the cosine differences as loss
            self.simnet_loss = tf.losses.cosine_distance(self.simnet_labels_ph, self.cosine_sim, axis = 1, weights=self.weights)

            # add regularization loss
            # self.simnet_loss += tf.reduce_mean(slim.losses.get_regularization_losses())

            self.simnet_loss = tf.add(self.simnet_loss,tf.reduce_mean(slim.losses.get_regularization_losses()),name='simnet_loss')

            # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
            learning_rate = tf.train.exponential_decay(SimnetConfig.LEARNING_RATE, self.global_step, 100, 0.80, staircase=True)

            # reduce the loss to optimize the network
            #optimizer = tf.train.MomentumOptimizer(learning_rate,momentum=0.95,use_nesterov=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)

            # create a train op: compute the loos, apply the gradient and return the loss value
            self.simnet_train_op = slim.learning.create_train_op(self.simnet_loss, optimizer)

            # visualize the distributions of activations coming off particular layers
            tf.summary.histogram('conv1_0', self.conv1_0_act)
            tf.summary.histogram('conv2_0', self.conv2_0_act)
            tf.summary.histogram('conv2_1', self.conv2_0_act)

            tf.summary.histogram('fc1', self.fc1_act)
            tf.summary.histogram('fc2', self.fc2_act)
            tf.summary.histogram('fc3', self.fc3_act)
            tf.summary.histogram('fc4', self.fc4_act)

            #  connect nodes to loss,auc and accuracy
            tf.summary.scalar('loss', self.simnet_loss)
            tf.summary.scalar('accuracy', self.simnet_accuracy)

            # combine summary nodes into one operation to run them
            self.simnet_summary_op = tf.summary.merge_all()

    def simnet_arg_scope(self, mode = False):
        '''
            Specify default arguments which will be passed to the layers in simnet.
            Input:  weigth_decay is the l2 regularization coefficient for fc weights
            Output: default arguments to the layers of the network.
        '''

        # weight regularizer
        w_regularizer = slim.l2_regularizer(SimnetConfig.WEIGHT_DECAY)

        def batch_norm_fn(x):
            return slim.batch_norm(x, scope = tf.get_variable_scope().name + "/bn")

        # enable.disable learning the batch normalization and dropout ops
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training = mode):

            # default arguments passed to the conv2d and fc layers
            with slim.arg_scope([slim.fully_connected, slim.conv2d],
                                weights_regularizer = w_regularizer,
                                activation_fn = tf.nn.leaky_relu,
                                normalizer_fn = batch_norm_fn,
                               ) as arg_scope:

                    return arg_scope

    # Deprecated
    # def create_sim_map(self):
    #     '''
    #         create the similarity/association map for each target
    #     '''
    #     feat_shape = self.unit_features1.get_shape().as_list()[1]
    #
    #     # Construct intermediate inputs only during Simnet Inference
    #     # if self.training_type == TrainingType.SimnetRun:
    #     #     with tf.variable_scope('map_construct_inputs'):
    #     #         self.t_feat1_ph = tf.placeholder(tf.float32, [None,feat_shape], name = 't_feat1_ph')
    #     #         self.t_feat2_ph = tf.placeholder(tf.float32, [None,feat_shape], name = 't_feat2_ph')
    #     #         self.w_bbox_t_ph = tf.placeholder(tf.float32, [None,1], name = 'w_bbox_t_ph')
    #     #         self.w_appear_t_ph = tf.placeholder(tf.float32, [None,1], name = 'w_appear_t_ph')
    #     #         # self.num_targets = tf.placeholder(tf.int32, name = 'num_targets')
    #     #         self.m_bbox_map = tf.placeholder(tf.float32, self.loc_shape + [512], 'm_bbox_map')
    #     #         self.m_appear_map = tf.placeholder(tf.float32, self.loc_shape + [512], 'm_appear_map')
    #     #         self.bbox_weight_map = tf.placeholder(tf.float32,self.loc_shape+[1], 'bbox_weight_map')
    #     #         self.appear_weight_map = tf.placeholder(tf.float32,self.loc_shape+[1], 'appear_weight_map')
    #     #         self.in_tar_cntr_xy = tf.placeholder(tf.float32, self.cntr_xy_shape, 'target_center_xy')
    #     # else:
    #     self.t_feat1_ph = self.t_feat1
    #     self.t_feat2_ph = self.t_feat2
    #     self.w_bbox_t_ph = self.w_bbox_t
    #     self.w_appear_t_ph = self.w_appear_t
    #
    #     with tf.variable_scope('map_construct'):
    #
    #         # threshold used to find where measurements are located
    #         cond_th = tf.constant(-1,tf.int32, name = 'condition_threshold')
    #         tt_feat1 = tf.transpose(self.t_feat1_ph, name ='feat_transpose1')
    #         tt_feat2 = tf.transpose(self.t_feat2_ph, name ='feat_transpose')
    #
    #         # target features
    #         kernels1 = tf.reshape(tt_feat1, [1,-1,1,self.num_targets], 'kernels1')
    #         kernels2 = tf.reshape(tt_feat2, [1,-1,1,self.num_targets], 'kernels2')
    #
    #         zeros = tf.constant(0, tf.float32, [feat_shape], name = 'zeros')
    #
    #         def maps(i):
    #             '''
    #              First, place measurement bbox and appearance features into appropriate bins
    #              Then, place branch weights computed for measurements into appropriate bins
    #             '''
    #             # global feature maps of measurements:
    #             global_map_feat1 = tf.cond(tf.equal(i,cond_th), lambda : zeros, lambda: self.m_feat1[i,:])
    #             global_map_feat2 = tf.cond(tf.equal(i,cond_th), lambda : zeros, lambda: self.m_feat2[i,:])
    #
    #             # branch-weight maps of measurements
    #             w_m1 = tf.cond(tf.equal(i,cond_th), lambda : 0.0 , lambda : self.w_bbox_m[i,0])
    #             w_m2 = tf.cond(tf.equal(i,cond_th), lambda : 0.0 , lambda : self.w_appear_m[i,0])
    #
    #             return global_map_feat1, global_map_feat2, w_m1, w_m2
    #
    #         if self.training_type in [TrainingType.Simnet, TrainingType.SimnetRun, TrainingType.Both]:
    #             self.m_bbox_map, self.m_appear_map, self.bbox_weight_map, self.appear_weight_map = tf.map_fn(maps, self.in_meas_loc_ph, dtype = (tf.float32,)*4)
    #
    #         # add batch and depth dimensions to the measurement map
    #         meas_map_batch1 = tf.expand_dims(self.m_bbox_map, axis = 0, name = 'batch_dim1')
    #         meas_map_batch2 = tf.expand_dims(self.m_appear_map, axis = 0, name = 'batch_dim2')
    #
    #         meas_map_out1 = tf.expand_dims(meas_map_batch1, axis = -1, name = 'depth1')
    #         meas_map_out2 = tf.expand_dims(meas_map_batch2, axis = -1, name = 'depth2')
    #
    #         # find correlation scores by using target features as kernels
    #         self.corr_map_bbox = tf.nn.convolution(meas_map_out1,
    #                                                kernels1,
    #                                                padding='VALID',
    #                                                strides=[1,1],
    #                                                name = 'sim_map1')
    #
    #         # clip cosine similarities to be sure they are in [-1,1]
    #         # self.corr_map_bbox = tf.clip_by_value(self.corr_map_bbox, -1.0, 1.0)
    #
    #         self.corr_map_appear = tf.nn.convolution(meas_map_out2,
    #                                                  kernels2,
    #                                                  padding='VALID',
    #                                                  strides=[1,1],
    #                                                  name = 'sim_map2')
    #
    #         # clip cosine similarities to be sure they are in [-1,1]
    #         self.corr_map_appear = tf.clip_by_value(self.corr_map_appear, -1.0, 1.0)
    #
    #         # reshape 1D corr_map to place a map for each target in channels
    #         shape_params = self.map_shape[0:] + [self.num_targets]
    #
    #         # swap channel by batch to place maps for each targets in the first dimension
    #         self.corr_map_bbox = tf.reshape(self.corr_map_bbox, shape_params)
    #         self.corr_map_bbox = tf.transpose(self.corr_map_bbox, [2, 0, 1])
    #
    #         self.corr_map_appear = tf.reshape(self.corr_map_appear, shape_params)
    #         self.corr_map_appear = tf.transpose(self.corr_map_appear, [2, 0, 1])
    #
    #         # reshape 1D measurement weight maps
    #         self.w_map1 = tf.reshape(self.bbox_weight_map,self.map_shape[0:])
    #         self.w_map2 = tf.reshape(self.appear_weight_map,self.map_shape[0:])
    #
    #         def weight_corr_maps(inputs):
    #
    #             # multiply target branch weights with those of measurements
    #             w_i_map1 = tf.multiply(inputs[0],self.w_map1) # bbox branch
    #             w_i_map2 = tf.multiply(inputs[1],self.w_map2) # appear branch
    #
    #             # normalization constant
    #             W_i = tf.add(w_i_map1,w_i_map2) + 1e-10
    #             w_i_map1 = tf.divide(w_i_map1,W_i)
    #             w_i_map2 = tf.divide(w_i_map2,W_i)
    #
    #             # compute weighted correlation maps for each branches
    #             weighted_corr_map1 = tf.multiply(w_i_map1, inputs[2])
    #             weighted_corr_map2 = tf.multiply(w_i_map2, inputs[3])
    #
    #             return weighted_corr_map1, weighted_corr_map2
    #
    #         # weight correlation maps by bbox and appearance branch weights for each targets
    #         self.corr_map_bbox, self.corr_map_appear = tf.map_fn(weight_corr_maps, [self.w_bbox_t_ph,
    #                                                                                 self.w_appear_t_ph,
    #                                                                                 self.corr_map_bbox,
    #                                                                                 self.corr_map_appear
    #                                                                                 ],
    #                                                              dtype=(tf.float32,tf.float32)
    #                                                              )
    #
    #         # combine correlation maps with weights of bbox and appearance branches
    #         # self.corr_map = tf.add(self.corr_map_bbox, self.corr_map_appear, 'corr_map')
    #         self.corr_map = tf.add(self.corr_map_bbox, self.corr_map_appear)
    #
    #         # swap batch by channel for padding to place the target index in the last dimension
    #         self.corr_map = tf.transpose(self.corr_map,[1,2,0])
    #
    #         # new dimensions for padding corr_maps
    #         new_height = int(self.map_shape[0] + (GlobalConfig.CROP_SIZE-1))
    #
    #         new_width =  int(self.map_shape[1] + (GlobalConfig.CROP_SIZE-1))
    #
    #         # pad correlation maps by adding zeros on top, on the left, and then on the
    #         # bottom and right until it has dimensions `target_height`, `target_width
    #         self.corr_map = tf.image.resize_image_with_crop_or_pad(self.corr_map, new_height, new_width)
    #
    #         # swap batch by channel to place target index are in the first dimension
    #         self.corr_map = tf.transpose(self.corr_map,[2,0,1])
    #
    #         # crop local correlation maps for targets using their centers
    #         def cropping(inputs):
    #
    #             # new target center after padding
    #             t_y = inputs[1][0,0] + GlobalConfig.CROP_SIZE//2
    #             t_x = inputs[1][0,1] + GlobalConfig.CROP_SIZE//2
    #
    #             # offset width
    #             x_top_left = tf.cast(t_x - GlobalConfig.CROP_SIZE//2 - 1,tf.int32)
    #             # offset height
    #             y_top_left = tf.cast(t_y - GlobalConfig.CROP_SIZE//2 - 1,tf.int32)
    #
    #             image = tf.expand_dims(inputs[0],axis=2)
    #
    #             local_map = tf.image.crop_to_bounding_box(image,
    #                                                       y_top_left,
    #                                                       x_top_left,
    #                                                       GlobalConfig.CROP_SIZE,
    #                                                       GlobalConfig.CROP_SIZE
    #                                                      )
    #
    #
    #             return local_map
    #
    #         self.local_corr_map = tf.map_fn(cropping, [self.corr_map,self.in_tar_cntr_xy], dtype= tf.float32)
    #
    #         # swap batch by channel to place target index in the last channel
    #         self.local_corr_map= tf.transpose(self.local_corr_map,[3,1,2,0])
    #
    #         def add_dummy_maps():
    #             map0_shape = self.local_corr_map.get_shape().as_list()[0:-1] +  \
    #                          [tf.subtract(self.max_target,self.num_targets)]
    #
    #             map_zeros = tf.zeros(map0_shape,tf.float32)
    #
    #             return tf.concat([self.local_corr_map,map_zeros],axis=3)
    #
    #         self.local_corr_map = tf.cond(tf.less(self.num_targets,self.max_target),
    #                                       lambda: add_dummy_maps(), lambda: self.local_corr_map)

    def assocnet_arg_scope(self):
        '''
            Specify default arguments which will be passed to the layers in simnet.
            Output: default arguments to the layers of the network.
        '''

        # weight regularizer
        w_regularizer = slim.l2_regularizer(AssocnetConfig.WEIGHT_DECAY)

        w_initializer = tf.variance_scaling_initializer(mode='fan_in')

        # default arguments passed to the conv2d and fc layers
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn = tf.nn.leaky_relu,
                            weights_regularizer = w_regularizer,
                            weights_initializer = w_initializer,
                            normalizer_fn = slim.batch_norm,
                           ) as arg_scope:

                return arg_scope

    def build_assoc_model(self, num_layers=3, num_kernels=32, kernel_size=3):
        '''
            Creates a fully convolutional network to predict the maps showing associated object.
        '''

        if self.training_type == TrainingType.Inference:
            with self.graph.as_default():
                saver = tf.train.import_meta_graph(self.args.assocnet_ckpt_path +'.meta')
                saver.restore(self.sess, self.args.assocnet_ckpt_path)
        else:
            with self.graph.as_default():

                self.in_local_corr_map_ph = tf.placeholder(tf.float32, shape = self.corr_map_shape, name = "local_corr_map")

                self.in_num_target_array_ph = tf.placeholder(tf.int32, shape = [1] ,name = "num_target_array")

                self.in_assoc_labels_ph = tf.placeholder(tf.float32, shape = self.label_shape, name="assoc_labels")

                self.in_avg_loss_ph = tf.placeholder(tf.float32, name = 'average_loss')

                self.in_avg_accuracy_ph = tf.placeholder(tf.float32, name = 'average_accuracy')

                self.in_exp_index_ph = tf.placeholder(tf.float32, name = 'example_index')

                self.in_global_step_assoc_ph = tf.placeholder(tf.float32, name = 'epoch_no')

                self.in_training_assocnet_ph = tf.placeholder(tf.bool, name = 'training')

                # ordered convolution activations
                hist_activations = OrderedDict()

                # input node for the first dilated convolutional layer
                net = self.in_local_corr_map_ph

                with slim.arg_scope( self.assocnet_arg_scope() ):

                    with tf.variable_scope('assocnet'):

                            # extract an abstract input feature maps from all local correlation maps
                            # net = slim.conv2d(self.in_local_corr_map_ph,1,1)

                            for l in range(0, num_layers):

                                # the dilation rate for convolutional layers
                                if (l < 2):
                                    dilation_rate = 2**(l+1)
                                else:
                                    dilation_rate = 6

                                net = slim.conv2d(net,num_kernels,kernel_size,rate=dilation_rate, padding='SAME',
                                                normalizer_params={'is_training': self.in_training_assocnet_ph,
                                                                    'scope': 'bn_{}'.format(str(l))
                                                                    },
                                                scope='conv_{}'.format(str(l)))

                                # save activations to summarize their histogram
                                hist_activations[l] = net

                            # Output Map
                            with tf.variable_scope('output_map'):

                                # mask for the relevant pixels
                                self.masks_pos = tf.cast(tf.greater(self.in_local_corr_map_ph, 0), tf.int64)

                                self.masks_neg = tf.cast(tf.less(self.in_local_corr_map_ph, 0), tf.int64)

                                # mask used for irrelevant pixels
                                self.masks =  1 - (self.masks_pos +  self.masks_neg)

                                # use this to make logits for irrelevant pixels -inf
                                self.masks = tf.to_float(np.finfo(np.float64).min*self.masks)

                                logits_maps = slim.conv2d(net, self.channels, kernel_size, scope='logits')

                                # make the network predict one of the measurements if available for each targets.
                                logits_maps = tf.add(logits_maps, self.masks, 'masked_logits')

                                # compute the logits for undetection probabilities of each targets
                                net= slim.flatten(logits_maps, scope='flat_logit_maps')

                                net = slim.fully_connected(net, 512, scope='fc1')

                                net = slim.fully_connected(net, 512, scope='fc2')

                                logits_undetect = slim.fully_connected(net, self.channels, scope='fc3')

                                # flat out maps for each target
                                logits_maps_1d = tf.reshape(logits_maps,[1,GlobalConfig.CROP_SIZE**2,-1])

                                # flat out undetection logits
                                logits_undetect_1d = tf.reshape(logits_undetect,[1, 1, -1])

                                # concatenate logits for maps and pixels for undetections
                                logits_map_undetect = tf.concat([logits_maps_1d,logits_undetect_1d], axis=1)

                                # perform softmax op on each map to compute association prob. and 1 - detection probability
                                maps_undetect_1d =tf.nn.softmax(logits_map_undetect, axis=1)

                                # obtain real maps for each targets
                                maps_1d = tf.slice(maps_undetect_1d, [0,0,0], [1, GlobalConfig.CROP_SIZE**2, -1])

                                # obtain detection probability for each targets
                                self.Pd = 1 - tf.slice(maps_undetect_1d, [0, GlobalConfig.CROP_SIZE**2, 0], [1,-1,-1])

                                # 2D predicted output maps
                                self.maps = tf.reshape(maps_1d,tf.shape(logits_maps))

                # histogram summaries
                with tf.name_scope("summaries"):
                        for k in hist_activations.keys():
                            tf.summary.histogram('conv_{0:d} activations'.format(k), hist_activations[k])

                # remove dummy maps, labels and Pd's
                self.process_maps_labels_Pd()

                with tf.name_scope('performance_measures'):

                    self.x_pred, self.y_pred = max_indices(self.sliced_maps, GlobalConfig.CROP_SIZE, GlobalConfig.CROP_SIZE)

                    self.x_true, self.y_true = max_indices(self.sliced_labels, GlobalConfig.CROP_SIZE, GlobalConfig.CROP_SIZE)

                    # validate predictions and gt pixel indices
                    self.get_relevant_indices()

                    # Identity to get a named op in the graph

                    self.x_pred = tf.identity(self.x_pred,name='x_pred')
                    self.y_pred = tf.identity(self.y_pred,name='y_pred')

                    self.correct_pred = correct_predictions(self.x_pred, self.y_pred, self.x_true,self.y_true)

                    self.accuracy = tf.reduce_mean(self.correct_pred, name='accuracy')

                    self.avg_accuracy = (self.in_avg_accuracy_ph*self.in_exp_index_ph + self.accuracy)/(self.in_exp_index_ph + 1)

                # compute the contrastive cross entropy loss
                self.loss = self.get_cross_entropy_loss()

                # average loss over epoches
                self.avg_loss = (self.in_avg_loss_ph*self.in_exp_index_ph + self.loss)/(self.in_exp_index_ph + 1)
                self.assoc_saver = tf.train.Saver()

                if self.training_type != TrainingType.Both and self.training_type != TrainingType.Assocnet:
                    self.assoc_saver.restore(self.sess, self.args.assocnet_ckpt_path)
                #else:
                    # self.in_local_corr_map_ph = self.local_corr_map
                    #self.in_num_target_array_ph = [self.num_targets]

    def get_relevant_indices(self):
        '''
            Find the relevant pixel indices where possible measurements are located for each targets.
        '''

        # loop index
        i = tf.constant(0)

        # indices for unassociated targets
        unassoc_indices = (np.iinfo(np.int32).min, np.iinfo(np.int32).min)

        loop_cond = lambda i, masks, x, y: tf.less(i, self.in_num_target_array_ph[0])

        def loop_body_gt(i, masks, x, y):
            # find validated x and y values
            x0,y0 = tf.cond(tf.equal(masks[0][x[0],y[0],i],0), lambda: unassoc_indices, lambda: (x[0],y[0]))

            # remove the first elements
            x = tf.slice(x,[1],[-1])
            y = tf.slice(y,[1],[-1])

            # append the new elements
            x = tf.concat([x,[x0]],axis=0)
            y = tf.concat([y,[y0]],axis=0)

            return tf.add(i, 1), masks, x, y

        def loop_body_pred(i, masks, x, y):

            # check if target is detected or not before associating it to a measurement
            undetect_cond = tf.less_equal(self.sliced_maps[0][x[0],y[0],i],1 - self.sliced_Pd[i])

            # validation condition for indices of maximum pixel
            val_cond = tf.equal(masks[0][x[0],y[0],i],0)

            # find validated x and y values
            x0,y0 = tf.cond(tf.logical_or(val_cond,undetect_cond), lambda: unassoc_indices, lambda: (x[0], y[0]))

            # remove the first elements
            x = tf.slice(x,[1],[-1])
            y = tf.slice(y,[1],[-1])

            # append the new elements
            x = tf.concat([x,[x0]],axis=0)
            y = tf.concat([y,[y0]],axis=0)

            return tf.add(i, 1), masks, x, y

        masks = tf.cast(self.masks_pos + self.masks_neg, tf.int32)

        # validate predictions is one of the nonzero pixels in local maps   and greater than 1 - Pd
        _, _,  self.x_pred, self.y_pred = tf.while_loop(loop_cond,loop_body_pred,[i, masks, self.x_pred, self.y_pred])

        # validate gt is one of the nonzero pixels in local maps
        _, _, self.x_true, self.y_true = tf.while_loop(loop_cond,loop_body_gt,[i, masks, self.x_true, self.y_true])

    def get_l2_norm_loss(self):
        '''

            Compute the loss between the predicted maps and label maps as the weighted L2
            norms and regularization term.
            :Return:
                avg_cost is the average of the L@ norms and regularization term.
        '''


        with tf.name_scope("l2_norm"):

            diff = tf.subtract(self.sliced_labels, self.sliced_maps)

            # do not penalize pixels if they are similar enough
            contrastive_diff = tf.maximum(tf.abs(diff) - 0.1 ,0)

            # L2 norm of contrastive differences
            l2_loss = tf.pow(contrastive_diff, 2)

            # weight individual loss for each maps
            # weighted_loss = tf.multiply(l2_loss, self.class_weights)

            # compute the l2 norms for each maps
            loss_map = tf.reduce_sum(l2_loss, axis=[1, 2, 3])
            # compute the average loss
            avg_loss = tf.reduce_mean(loss_map)

            # regularization term
            reg_terms = tf.losses.get_regularization_losses(scope='assocnet')

            avg_loss += AssocnetConfig.WEIGHT_DECAY* tf.reduce_sum(reg_terms)

            return avg_loss

    def get_cross_entropy_loss(self):
        '''
            Compute the loss between the predicted maps and label maps as the sum of
            the weighted cross entropies of each elements and regularization term.
            :Return:
                avg_cost is the average of the weighted cross entropies and regularization
                term.
        '''
        def elemwise_cross_entropy(x):
            '''
                x[0]: element from the label map.
                x[1]: element from the predicted map.
                x[2]: weight constant.
            '''

        with tf.name_scope("cross_entropy"):

            # compute 1 - p for label 0 and p for label 1
            diff =  tf.abs(tf.abs(self.sliced_labels - 1.0) - self.sliced_maps)

            # do not penalize pixels if they are similar enough
            contrastive_diff = tf.minimum( diff + 0.01, 1.0)

            cross_entropies = -tf.log(contrastive_diff)

            # compute element-wise losses
            #weighted_loss = tf.multiply(cross_entropies, self.class_weights)

            # compute the loss for each maps
            loss_map = tf.reduce_sum(cross_entropies, axis=[1, 2, 3])

            # compute the average loss
            avg_loss = tf.reduce_mean(loss_map)

            # regularization term
            reg_terms = tf.losses.get_regularization_losses(scope='assocnet')

            avg_loss += AssocnetConfig.WEIGHT_DECAY*tf.reduce_sum(reg_terms)

            # compute logits for detection probabilities in the range [-20,20] for [0,1]
            logits_Pd = tf.log((self.sliced_Pd + 10**-20) / (1 - self.sliced_Pd + 10**-20))

            # use self.x_true or self.y_true to check if each target is detected.
            self.true_Pd = tf.cast(tf.not_equal(self.x_true,np.iinfo(np.int32).min), tf.float32)

            # since missed detections are sparse, weight their losses more
            weights = self.args.skewness_undetection*(1 - self.true_Pd) + self.true_Pd

            # compute cross entropy losses for probability of detections
            self.losses_Pd = tf.nn.weighted_cross_entropy_with_logits(targets=self.true_Pd,
                                                                      logits=logits_Pd,
                                                                      pos_weight= weights,
                                                                      name='detection_losses')

            avg_loss = tf.add(avg_loss, tf.reduce_mean(self.losses_Pd), name='avg_loss')

        return avg_loss

    def process_maps_labels_Pd(self):
        '''
            Remove dummy channels from the labels, the predicted output_maps and probability detections.
        '''

        def map_slice(x):
            y = tf.slice(x[0], [0, 0, 0], [GlobalConfig.CROP_SIZE, GlobalConfig.CROP_SIZE, x[1]])
            return y

        def Pd_slice(x):
            y = tf.slice(x[0], [0, 0], [1, x[1]])
            return y

        # output maps predicted for each existing targets.
        self.sliced_maps = tf.map_fn(map_slice, (self.maps, tf.reshape(self.in_num_target_array_ph,[1])), dtype=tf.float32)
        #label maps for each existing targets.
        self.sliced_labels = tf.map_fn(map_slice, (self.in_assoc_labels_ph, tf.reshape(self.in_num_target_array_ph,[1])), dtype=tf.float32)
        # probability of detection for each existing targets
        self.sliced_Pd = tf.map_fn(Pd_slice, (self.Pd, tf.reshape(self.in_num_target_array_ph,[1])), dtype=tf.float32)

        # flat out logits_Pd for each existing target
        self.sliced_Pd = tf.reshape(tf.squeeze(self.sliced_Pd),[-1],name = 'detection_probabilities')

        # weight loss considering skewness of zeros and ones in label maps
        num_pixels = GlobalConfig.CROP_SIZE**2
        self.class_weights = tf.add(tf.multiply(self.sliced_labels, (num_pixels - 2)/num_pixels), 1/num_pixels)

    def reset_graph(self):
        self.sess.close()

        '''reset default graph'''
        # tf.reset_default_graph()
        self.graph = tf.Graph()
        self.sess = tf.Session(config=AssocModel.config, graph=self.graph)

"""AVOD VGG Network."""

import os
import tensorflow as tf
import fan_track.avod_feature_extractor.avod_bev_vgg_graph as avod_bev_vgg
slim = tf.contrib.slim
framework = tf.contrib.framework
from avod import root_dir as avod_root_dir
from avod.builders.dataset_builder import DatasetBuilder
import avod.builders.config_builder_util as config_builder
from avod.core import anchor_projector
from avod.core import constants

import numpy as np

from avod.core import box_3d_encoder

class BevVGGNet():

    def __init__(self, avod_model_path):

        # Create local graph and use it in the session
        self.g = tf.Graph()

        # feature maps from the bev_vgg
        self.feat_maps = None

        # Specify where the avod model was saved.
        self.avod_model_path = avod_model_path

        root_dir = os.path.dirname(os.path.abspath(__file__))

        # specify where the new model will live:
        vgg_checkpoint_path = root_dir  + "/checkpoints/bev_vgg"

        self.avod_file = tf.train.latest_checkpoint(self.avod_model_path)

        self.vgg_ckpt_file = vgg_checkpoint_path + "/model.ckpt"

        if not tf.gfile.Exists(vgg_checkpoint_path):
            tf.gfile.MakeDirs(vgg_checkpoint_path)

    def setup(self, bev_pixel_size, bev_depth, model_config, roi_crop_size, bev_extents):
        '''construction phase that assembles the bev_vgg graph'''

        bev_feature_extractor = avod_bev_vgg.BevVgg(model_config)

        # Create the graph
        with self.g.as_default():

            # dummy nodes that provide entry points for input image to the graph
            bev_dim = np.append(bev_pixel_size,bev_depth)
            with tf.variable_scope('bev_input'):
                bev_input_placeholder = tf.placeholder(tf.float32, bev_dim,"in")
                bev_input_batches = tf.expand_dims(bev_input_placeholder, axis=0)

                # the preprocessed tensor object
                bev_preprocessed = bev_feature_extractor.preprocess_input(bev_input_batches,bev_pixel_size)

            # define the bev_vgg model
            bev_feature_maps, self.end_points = bev_feature_extractor.build(bev_preprocessed,bev_pixel_size,False)
            self.end_points['bev_vgg/upsampling/maps_out'] = bev_feature_maps

            # Extract ROIs
            bev_rois, bev_pred_boxes = self.RoIPool(bev_feature_maps,roi_crop_size,bev_extents)

            self.end_points['roi_pooling/crop_resize/bev_rois'] = bev_rois
            self.end_points['roi_pooling/crop_resize/bev_pred_boxes'] = bev_pred_boxes

    def get_box_indices(self,boxes):
                        proposals_shape = boxes.get_shape().as_list()
                        if any(dim is None for dim in proposals_shape):
                            proposals_shape = tf.shape(boxes)
                        ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
                        multiplier = tf.expand_dims(
                            tf.range(start=0, limit=proposals_shape[0]), 1)
                        return tf.reshape(ones_mat * multiplier, [-1])

    def RoIPool(self,bev_feature_maps,roi_crop_size,bev_extents):

        with tf.variable_scope('roi_pooling'):
            # entry points for predicted bboxes in the in the format [x, y, z, dim_x, dim_y, dim_z]
            pred_placeholder = tf.placeholder(tf.float32,[None,6],'predictions')

            bev_extents= tf.constant(bev_extents, dtype=tf.float32)


            with tf.variable_scope('crop_resize'):

                    # project 3d predicted bboxes onto the image place
                    bev_pred_boxes, bev_pred_boxes_norm = anchor_projector.project_to_bev(pred_placeholder,bev_extents)


                    # reorder the projected bboxes for crop and resizing
                    bev_pred_boxes_norm_tf_order = anchor_projector.reorder_projected_boxes(bev_pred_boxes_norm)

                    bev_feature_maps = self.end_points['bev_vgg/upsampling/maps_out']

                    bev_boxes_norm_batches = tf.expand_dims(bev_pred_boxes_norm, axis=0)

                    # These should be all 0's since there is only 1 image since box_indices[i] specifies image
                    box_indices = self.get_box_indices(bev_boxes_norm_batches)

                    # Do ROI Pooling on BEV
                    bev_rois = tf.image.crop_and_resize(bev_feature_maps, bev_pred_boxes_norm_tf_order,box_indices,roi_crop_size, name='bev_roi')

                    return bev_rois, bev_pred_boxes


    def restore(self):
        '''restore bev_vgg from the avod's checkpoint'''

        with tf.Session(graph = self.g) as sess:
            # Get all variables,i.e. only those created in the bev_vgg to restore
            variables_to_restore = slim.get_variables_to_restore()

            # Restore only the convolutional layers from the avod checkpoint
            init_fn = framework.assign_from_checkpoint_fn(self.avod_file, variables_to_restore)
            # create the saver object to restore the variables of the bev_vgg
            saver = tf.train.Saver()

            # restore the bev_vgg's variables from the avod's checkpoint file
            if not(init_fn is None):
                with tf.Session(graph=self.g) as sess:
                    # perform the assignment, i.e.,restore
                    init_fn(sess)
                    saver.save(sess, self.vgg_ckpt_file)
                    print("bev_vgg checkpoint file was saved.")

    def restore_graph(self):

        # sess will launch/handle img_vgg graph until
        self.sess = tf.Session(graph=self.g)

        # make the img_vgg the default graph
        with self.g.as_default():
            # create a saver object to restore the variables
            saver = tf.train.Saver()
            # Restore variables from disk
            saver.restore(self.sess, self.vgg_ckpt_file)

    def close_session(self):
        '''close the session running the bev_vgg graph'''
        if self.sess is not None:
            self.sess.close()

    def exract_features(self,input_bev,predictions):
        '''launch the bev_graph to extract bev features'''

        # access the input image placeholder variable
        bev_input_placeholder = self.g.get_tensor_by_name("bev_input/in:0")
        #self.g.get_tensor_by_name("bev_vgg/upsampling/maps_out:0")

        pred_placeholder = self.g.get_tensor_by_name("roi_pooling/predictions:0")

        # access the op that you want to run.
        feature_maps_out = self.end_points['bev_vgg/upsampling/maps_out']

        bev_rois = self.end_points['roi_pooling/crop_resize/bev_rois']

        bev_pred_boxes = self.end_points['roi_pooling/crop_resize/bev_pred_boxes']

        # return the feature maps from the bev_vgg graph
        feat_maps, obj_rois, obj_pred_boxes  = self.sess.run([feature_maps_out, bev_rois,bev_pred_boxes],
                                                             feed_dict = {bev_input_placeholder:input_bev,
                                                                          pred_placeholder: predictions})

        # print("bev outputs were obtained.")

        return feat_maps, obj_rois, obj_pred_boxes


    def checkRestore(self):
        """Check variables are correctly restored to bev_vgg"""

        w_bev_vgg = []

        # launch the bev_vgg graph
        with tf.Session(graph=self.g) as sess:
            saver = tf.train.Saver()

            # Restore variables from disk.
            saver.restore(sess, self.vgg_ckpt_file)

            for v in tf.trainable_variables():
                if (v.name.find("bev_vgg") != - 1):
                    if (v.name.find("weights")  != -1):
                        w_bev_vgg.append(sess.run(v))
                    elif (v.name.find("bias")  != -1):
                        w_bev_vgg.append(sess.run(v))
                    elif (v.name.find("beta")  != -1):
                        w_bev_vgg.append(sess.run(v))
                    elif (v.name.find("gamma")  != -1):
                        w_bev_vgg.append(sess.run(v))

        w_avod_bev_vgg = []

        # launch the avod's graph
        avod_graph = tf.Graph()
        with tf.Session(graph=avod_graph) as sess:
            saver = tf.train.import_meta_graph(self.avod_file + ".meta")

            saver.restore(sess, self.avod_file)

            for v in tf.trainable_variables():
                if (v.name.find("bev_vgg") != - 1):
                    if (v.name.find("weights")  != -1):
                        w_avod_bev_vgg.append(sess.run(v))
                    elif (v.name.find("bias")  != -1):
                        w_avod_bev_vgg.append(sess.run(v))
                    elif (v.name.find("beta")  != -1):
                        w_avod_bev_vgg.append(sess.run(v))
                    elif (v.name.find("gamma")  != -1):
                        w_avod_bev_vgg.append(sess.run(v))


        for i,w in enumerate(w_bev_vgg):
            if not(np.array_equal(w,w_avod_bev_vgg[i])):
                print("resotoring bev_vgg failed.")
                return False;

        # print("bev_vgg restored successfully.")
        return True

if __name__ == '__main__':

    # checkpoint name
    # ckpt_name = 'cars_reduced_size'
    ckpt_name = 'pyramid_cars_with_aug_example'

    exp_config = ckpt_name + '.config'

    experiment_config_path = avod_root_dir() + '/data/outputs' + \
                                               '/' + ckpt_name + \
                                               '/' + exp_config

    # Read the configurations
    model_config, _, _, dataset_config = config_builder.get_configs_from_pipeline_file(
                                              experiment_config_path, is_training=False)

    # Overwrite the defaults
    dataset_config = config_builder.proto_to_obj(dataset_config)

    dataset_config.data_split = 'test'
    dataset_config.data_split_dir = 'testing'
    dataset_config.has_labels = False
    # Remove augmentation during evaluation in test mode
    dataset_config.aug_list = []

    # Build the kitti dataset object
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,use_defaults=False)

    # a particular sample index in the lidar point cloud dataset
    sample_index = None

    # For testing, any sample should work
    if sample_index is not None:
        samples = dataset.load_samples([sample_index])
    else:
        samples = dataset.next_batch(batch_size=1, shuffle=False)

    # Only handle one sample at a time for now
    sample = samples[0]

    bev_image = sample.get(constants.KEY_BEV_INPUT)

    bev_vgg_config = model_config.layers_config.bev_feature_extractor.bev_vgg

    bev_pixel_size = np.asarray(bev_image.shape[:2])

    bev_depth = model_config.input_config.bev_depth

    bev_extents = dataset.kitti_utils.bev_extents

    roi_crop_size = [model_config.avod_config.avod_proposal_roi_crop_size] * 2

    avod_model_path = model_config.paths_config.checkpoint_dir

    bev_vgg = BevVGGNet(avod_model_path)

    bev_vgg.setup(bev_pixel_size,bev_depth,bev_vgg_config,roi_crop_size,bev_extents)

    bev_vgg.restore()

    # predicted 3d bounding boxes in the format of  [x, y, z, l, w, h, ry]
    predictions = np.asarray([-6.037999, 2.202901, 23.712843, 3.576750, 1.555003,1.527328, 1.587202], dtype=np.float32)

    # convert 3d bbox predictions to anchors
    anchors = box_3d_encoder.box_3d_to_anchor(predictions)

    if (bev_vgg.checkRestore()):
        bev_vgg.exract_features(bev_image, anchors)

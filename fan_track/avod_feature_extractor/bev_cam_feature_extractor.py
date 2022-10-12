import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from fan_track.avod_feature_extractor.bev_vgg import BevVGGNet
from fan_track.avod_feature_extractor.img_vgg import ImgVGGNet
import avod.builders.config_builder_util as config_builder
from avod import root_dir as avod_root_dir
from avod.core import constants
from avod.core.avod_fc_layers import avod_fc_layer_utils
from avod.core import box_3d_encoder
from avod.core import box_3d_projector

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###################################
# Keys
###################################
CAM_IMG = 'camera_image'
BEV_IMG = 'bev_image'
IMG_PS = 'image_pixel_size'
BEV_PS = 'bev_pixel_size'
IMG_DPT = 'image_depth'
BEV_DPT = 'bev_depth'
BEV_VGG_CONFIG = 'bev_vgg_config'
IMG_VGG_CONFIG = 'img_vgg_config'
BEV_EXT = 'bev_extents'
ROI_SZ = 'roi_crop_size'
PRED_BBOX = 'prediction_bboxes'
STEREO_CALIB = 'stereo_calib'


class BevCamFeatExtractor(object):

    def __init__(self, dataset, model_config, rebuild_dataset = True):
        '''
        Read the avod's model configuration and prepare the test dataset.
        '''

        # Overwrite repeated field
        self.model_config = config_builder.proto_to_obj(model_config)
        # Switch path drop off during evaluation
        self.model_config.path_drop_probabilities = [1.0, 1.0]

        # Specify where the avod model was saved.
        self.avod_model_path = self.model_config.paths_config.checkpoint_dir

        # Build the kitti dataset according to given data configuration
        self.dataset = dataset

    def rebuild_dataset(self, dataset):
        # Build the kitti dataset according to given data configuration
        self.dataset = dataset

    def get_sample(self,sample_index = None):
        '''
         Retrieve the image, bev, ground plane and others used as inputs to the avod.
        Args:
            sample_index: a particular index of the testing set.
        '''

        # For testing, any sample should work
        if sample_index is not None:
            samples = self.dataset.load_samples([sample_index])
        else:
            samples = self.dataset.next_batch(batch_size=1,  shuffle=False)

        # only handle one sample at a time for
        self.sample = samples[0]

        self.inputs[CAM_IMG] = self.sample.get(constants.KEY_IMAGE_INPUT)

        self.inputs[STEREO_CALIB] = self.sample.get(constants.KEY_STEREO_CALIB_P2)

        # self.inputs[BEV_IMG] = self.sample.get(constants.KEY_BEV_INPUT)

    def prepare_inputs(self):
        '''Prepare default inputs to the detector avod'''

        self.inputs = {}

        bev_pixel_size = np.asarray([self.model_config.input_config.bev_dims_h,
                                     self.model_config.input_config.bev_dims_w])

        img_pixel_size = np.asarray([self.model_config.input_config.img_dims_h,
                                     self.model_config.input_config.img_dims_w])

        # inputs for img_vgg
        self.inputs[ROI_SZ] = [self.model_config.avod_config.avod_proposal_roi_crop_size] * 2

        self.inputs[IMG_DPT] = self.model_config.input_config.img_depth

        self.inputs[IMG_PS] = img_pixel_size

        self.inputs[IMG_VGG_CONFIG] = self.model_config.layers_config.img_feature_extractor.img_vgg


        # inputs for bev_vgg
        self.inputs[BEV_DPT] = self.model_config.input_config.bev_depth

        self.inputs[BEV_PS] = bev_pixel_size

        self.inputs[BEV_VGG_CONFIG] = self.model_config.layers_config.bev_feature_extractor.bev_vgg_pyr

        self.inputs[BEV_EXT] = self.dataset.kitti_utils.bev_extents

        #projected_bbox = box_3d_projector.project_to_image_space(predictions, self.inputs[STEREO_CALIB], image_shape = self.inputs[IMG_PS][::-1])

    def get_anchors(self,predictions):

        # convert 3d bbox predictions to anchors
        self.anchors = box_3d_encoder.box_3d_to_anchor(predictions)

        self.inputs[PRED_BBOX] = self.expand_predictions()

    def setup_bev_img_vggs(self):
        '''Constructs the bev and img vgg nets'''


        # self.bev_feat_extractor = bev_vgg.BevVGGNet(self.avod_model_path)
        # self.bev_feat_extractor.setup(self.inputs[BEV_PS],
        #                               self.inputs[BEV_DPT],
        #                               self.inputs[BEV_VGG_CONFIG],
        #                               self.inputs[ROI_SZ],
        #                               self.inputs[BEV_EXT])
        # self.bev_feat_extractor.restore()
        # self.bev_feat_extractor.restore_graph()

        self.img_feat_extractor = ImgVGGNet(self.avod_model_path)
        self.img_feat_extractor.setup(self.inputs[IMG_PS],
                                      self.inputs[IMG_DPT],
                                      self.inputs[IMG_VGG_CONFIG],
                                      self.inputs[ROI_SZ])
        # restore the graph variables from the checkpoint file
        self.img_feat_extractor.restore_graph()

    def extract_features(self):
        '''Extracts the bev and img feature maps. object rois, and projection of 3D bboxes onto image planes'''

        # self.bev_feat_map,self.obj_bev_rois,self.obj_bev_bboxes = self.bev_feat_extractor.exract_features(self.inputs[BEV_IMG],
        #                                                                                              self.inputs[PRED_BBOX])

        self.img_feat_map,self.obj_img_rois,self.obj_img_bboxes = self.img_feat_extractor.extract_features(self.inputs[CAM_IMG],
                                                                                                      self.inputs[PRED_BBOX],
                                                                                                      self.inputs[STEREO_CALIB])
        # return self.obj_bev_rois, self.obj_img_rois
        return self.obj_img_rois

    def setup_expand_predictions(self):
        '''Expand anchors along x and z axes'''

        with tf.variable_scope('avod_projection'):

                # tf_anchors = tf.constant(self.anchors, tf.float32, name = 'anchors')
                anchors = tf.placeholder(tf.float32, [None,6], 'anchors')

                if (self.model_config.expand_proposals_xz > 0.0):

                    expand_length = self.model_config.expand_proposals_xz

                    with tf.variable_scope('expand_xz'):
                        expanded_dim_x = anchors[:, 3] + expand_length
                        expanded_dim_z = anchors[:, 5] + expand_length

                        self.avod_projection_in = tf.stack([
                                                            anchors[:, 0],
                                                            anchors[:, 1],
                                                            anchors[:, 2],
                                                            expanded_dim_x,
                                                            anchors[:, 4],
                                                            expanded_dim_z
                                                           ], axis=1)

                else:
                    self.avod_projection_in = anchors

    def expand_predictions(self):

        anchors = tf.get_default_graph().get_tensor_by_name('avod_projection/anchors:0')

        # run the  graph, i.e., evaluate the tensor avod_projection in:
        with tf.Session() as sess:
            ex_anchors = sess.run(self.avod_projection_in,feed_dict = {anchors:self.anchors})

        return ex_anchors

    def setup_early_fusion(self):
        '''
        Fuse the img and bev features according to the fusion method indicated in the model configuration
        Returns: the feature map after fusion of bev_rois and img_rois maps.
        """
        '''

        with tf.variable_scope('early_fusion'):

            # since path_drop_probabilities are 1 for evaluation
            bev_mask = tf.constant(1.0)
            img_mask = tf.constant(1.0)

            #img_roi_channels = self.model_config.layers_config.img_feature_extractor.img_vgg.vgg_conv4[1]
            #bev_roi_channels = self.model_config.layers_config.bev_feature_extractor.bev_vgg.vgg_conv4[1]

            # entry points for bev and img rois
            bev_rois = tf.placeholder(tf.float32,[None, None, None, None],'bev_rois')
            img_rois = tf.placeholder(tf.float32,[None, None, None, None],'img_rois')

            #bev_rois = tf.constant(self.obj_bev_rois)
            #img_rois = tf.constant(self.obj_img_rois)

            # fusion_method: 'mean' or 'concat'
            avod_fusion_method = self.model_config.layers_config.avod_config.fusion_fc_layers.fusion_method

            # fusion of features for rois
            self.fused_features = avod_fc_layer_utils.feature_fusion(fusion_method = avod_fusion_method,
                                                                     inputs=[bev_rois, img_rois],
                                                                     input_weights=[bev_mask, img_mask])

    def early_fusion(self):

        bev_rois = tf.get_default_graph().get_tensor_by_name('early_fusion/bev_rois:0')
        img_rois = tf.get_default_graph().get_tensor_by_name('early_fusion/img_rois:0')

        with tf.Session() as sess:
            self.fused_feat_out = sess.run(self.fused_features, feed_dict = {bev_rois: self.obj_bev_rois,
                                                                             img_rois: self.obj_img_rois})

        # print('fused features were obtained.')

        return self.fused_feat_out

    def close_sessions(self):
        '''
            clear the the default graph and close all sessions.
        '''

        self.img_feat_extractor.close_session()
        # self.bev_feat_extractor.close_session()

        #tf.reset_default_graph()

    def visualize(self,predictions):
        '''Display camera, bev images, the corresponding feature maps and rois'''

        # feature map index. It should be less than 255
        fidx = 200

        # display camera image, one of the img feature maps and the cropped rois on that feature map
        fig1 = plt.figure()

        ax = fig1.add_subplot(211)
        ax.imshow(self.inputs[CAM_IMG])
        ax.set_title('camera img to the feature extractor')
        ax.axis('off')

        obj_img_bboxes = box_3d_projector.project_to_image_space(predictions[0], self.inputs[STEREO_CALIB])

        # draw bboxes which are in the format of  [x1, y1, x2, y2] where N is the batch_size
        width  = round(obj_img_bboxes[2] - obj_img_bboxes[0])
        height = round(obj_img_bboxes[3] - obj_img_bboxes[1])

        #top left corner of the 2d bbox
        llc = (round(obj_img_bboxes[0]), round(obj_img_bboxes[1]))

        ax.add_patch( patches.Rectangle(llc, width, height, edgecolor='r', linewidth=1, fill=False) )

        ax = fig1.add_subplot(212)
        ax.imshow(self.img_feat_map[0,:,:,fidx])
        ax.set_title('camera img feature map from the feature extractor')
        ax.axis('off')

        # display bev image, one of the bev feature maps
        fig2 = plt.figure()

        ax = fig2.add_subplot(211)
        ax.imshow(self.inputs[BEV_IMG][:,:,0])
        ax.set_title('bev image to the feature extractor')
        ax.axis('off')

        obj_bev_bboxes,_ =  box_3d_projector.project_to_bev(predictions,self.inputs[BEV_EXT])

        # draw bboxes which are in the format of  N x [x1, y1, x2, y2] where N is the batch_size
        width  = round(obj_bev_bboxes[0][2,0] - obj_bev_bboxes[0][0,0])
        height = round(obj_bev_bboxes[0][0,1] - obj_bev_bboxes[0][2,1])

        #top left corner of the 2d bbox
        llc = (round(obj_bev_bboxes[0][0,0] + self.inputs[BEV_EXT][0,1]),
               round(self.inputs[BEV_EXT][1,1] - obj_bev_bboxes[0][0,1]))

        ax.add_patch( patches.Rectangle(llc, width, height,  edgecolor='w', fill=False) )

        ax = fig2.add_subplot(212)
        ax.imshow(self.bev_feat_map[0,:,:,fidx])
        ax.set_title('bev image map from the feature extractor')
        ax.axis('off')

        # display cropped img_roi, bev_roi and the fused roi
        fig3 = plt.figure()

        ax = fig3.add_subplot(311)
        ax.imshow(self.obj_img_rois[0,:,:,fidx])
        ax.set_title('camera img roi from the feature extractor')
        ax.axis('off')

        ax = fig3.add_subplot(312)
        ax.imshow(self.obj_bev_rois[0,:,:,fidx])
        ax.set_title('bev roi from the feature extractor')
        ax.axis('off')

        ax = fig3.add_subplot(313)
        ax.imshow(self.fused_feat_out[0,:,:,fidx])
        ax.set_title('the fused roi from the feature extractor')
        ax.axis('off')


        plt.show()

def main(_):

    # predicted 3d bounding boxes in the format of  [x, y, z, l, w, h, ry]
    predictions = np.asarray([[3.388173, 1.539172, 5.048745, 4.101504, 1.510562,1.381664, -1.576586]], dtype=np.float32)

    # read the configurations from the given config file and prepare the Kitti dataset
    feat_ext = BevCamFeatExtractor()

    # prepare the avod
    feat_ext.prepare_inputs()

    # construct bev_vgg and img_vgg nets
    feat_ext.setup_bev_img_vggs()

    # feat_ext.setup_early_fusion()

    feat_ext.setup_expand_predictions()

    # enter frame number number to get a training example together with other required inputs
    feat_ext.get_sample(1)

    feat_ext.get_anchors(predictions)

    # extract bev and cam features using vggs
    obj_bev_rois, obj_img_rois = feat_ext.extract_features()

    # output for classification
    # feat_ext.early_fusion()

    # feat_ext.visualize(predictions)

    feat_ext.close_sessions()

if __name__ == '__main__':
    tf.app.run()

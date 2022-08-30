"""
    Contains modified VGG model definition to extract features from
    RGB image input.
Usage:
    outputs, end_points = ImgVgg(inputs, layers_config)
"""
import tensorflow as tf
from fan_track.avod_feature_extractor.bev_feature_extractor import BevFeatureExtractor

slim = tf.contrib.slim

class BevVgg(BevFeatureExtractor):

    def vgg_arg_scope(self, weight_decay=0.0005):
        """Defines the VGG arg scope.
        Args:
           weight_decay: The l2 regularization coefficient.
        Returns:
          An arg_scope.
        """
        # activation function and normalization operation are shared by the all layers
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(
                                weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            #zero padding is applied to preserve the input_size at the output of the conv layers
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def build(self, inputs, input_pixel_size, is_training, scope='bev_vgg_pyr'):
        """ Modified VGG for BEV feature extraction with pyramid features
        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False for validation/testing.
            scope: Optional scope for the variables.
        Returns:
            The last op containing the log predictions and end_points dict.
        """
        vgg_config = self.config

        with slim.arg_scope(self.vgg_arg_scope(
                weight_decay=vgg_config.l2_weight_decay)):
            with tf.variable_scope(scope, 'bev_vgg_pyr', [inputs]) as sc:
                end_points_collection = sc.name + '_end_points'
                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    outputs_collections=end_points_collection):
                    # Pad 700 to 704 to allow even divisions for max pooling
                    padded = tf.pad(inputs, [[0, 0], [4, 0], [0, 0], [0, 0]])

                    # Encoder
                    conv1 = slim.repeat(padded,
                                        vgg_config.vgg_conv1[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv1[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv1')
                    pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')

                    conv2 = slim.repeat(pool1,
                                        vgg_config.vgg_conv2[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv2[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv2')
                    feat_map2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')

                    conv3 = slim.repeat(feat_map2,
                                        vgg_config.vgg_conv3[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv3[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv3')
                    pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')

                    feat_map4 = slim.repeat(pool3,
                                        vgg_config.vgg_conv4[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv4[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv4')

                    with tf.variable_scope('upsampling'):
                        # This extractor downsamples the input by a factor # of 8 (3 maxpool layers)
                        downsampling_factor = 8

                        downsampled_shape = input_pixel_size / downsampling_factor

                        # upsampled_shape = downsampled_shape * vgg_config.upsampling_multiplier
                        upsampled_shape = downsampled_shape * 2

                        # 4x bilinear upsampling layer. feature_maps_out is under the 'img_vgg\upsampling' scope
                        feat_map4 = tf.image.resize_bilinear(feat_map4, [176,200], name='feat_map4')

                        feature_maps_out = tf.concat((feat_map2, feat_map4), axis=3, name='maps_out')

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                return feature_maps_out, end_points

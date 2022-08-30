"""
    Contains modified VGG model definition to extract features from
    RGB image input.
Usage:
    outputs, end_points = ImgVgg(inputs, layers_config)
"""
import tensorflow as tf
from fan_track.avod_feature_extractor.img_feature_extractor import ImgFeatureExtractor

slim = tf.contrib.slim

class ImgVgg(ImgFeatureExtractor):

    def vgg_arg_scope(self):
        """Defines the VGG arg scope.
        Returns:
          An arg_scope.
        """
        # activation function and normalization operation are shared by the all layers
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu):
            #zero padding is applied to preserve the input_size at the output of the conv layers
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def build(self, inputs, input_pixel_size,is_training,scope='img_vgg'):
        """ Build the modified VGG for image feature extraction.
        Note: All the fully_connected layers have been transformed to conv2d
              layers and are implemented in the main model.
        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False for validation/testing.
            scope: Optional scope for the variables.
        Returns:
            The last op containing the log predictions and end_points dict.
        """
        vgg_config = self.config

        with slim.arg_scope(self.vgg_arg_scope()):
            with tf.variable_scope('vgg_16', [inputs]) as sc:
                end_points_collection = sc.name + '_end_points'
                # Collect outputs for conv2d, and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    outputs_collections=end_points_collection):
                    # 1st conv layer
                    net = slim.repeat(inputs,
                                      # vgg_config.vgg_conv1[0], #repetitions
                                      2, #repetitions
                                      slim.conv2d,             #type of the layer
                                      # vgg_config.vgg_conv1[1], #number of filters
                                      64, #number of filters
                                      [3, 3],                  #kernel size
                                      normalizer_params={
                                     'is_training': is_training},
                                      scope='conv1')
                    # add the max_pooling layer between
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    # 2nd conv layer
                    net = slim.repeat(net,
                                      # vgg_config.vgg_conv2[0],
                                      2,
                                      slim.conv2d,
                                      # vgg_config.vgg_conv2[1],
                                      128,
                                      [3, 3],
                                      normalizer_params={
                                          'is_training': is_training},
                                      scope='conv2')
                    feat_map2 = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(feat_map2,
                                      # vgg_config.vgg_conv3[0],
                                      3,
                                      slim.conv2d,
                                      # vgg_config.vgg_conv3[1],
                                      256,
                                      [3, 3],
                                      normalizer_params={
                                          'is_training': is_training},
                                      scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    feat_map4 = slim.repeat(net,
                                      # vgg_config.vgg_conv4[0],
                                      3,
                                      slim.conv2d,
                                      # vgg_config.vgg_conv4[1],
                                      512,
                                      [3, 3],
                                      normalizer_params={
                                          'is_training': is_training},
                                      scope='conv4')

                    net = slim.conv2d(feat_map4, 4096, [7, 7], padding='VALID', scope='fc6')
                    net = slim.dropout(net, 0.5, scope='dropout6')
                    net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                    net = slim.conv2d(net, 1000, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='fc8')

            with tf.variable_scope(scope, 'img_vgg', [inputs]) as sc:
                with tf.variable_scope('upsampling'):
                    # This extractor downsamples the input by a factor # of 8 (3 maxpool layers)
                    downsampling_factor = 8

                    downsampled_shape = input_pixel_size / downsampling_factor

                    # upsampled_shape = downsampled_shape * vgg_config.upsampling_multiplier
                    upsampled_shape = downsampled_shape * 2

                    # 4x bilinear upsampling layer. feature_maps_out is under the 'img_vgg\upsampling' scope
                    feat_map4 = tf.image.resize_bilinear(feat_map4, upsampled_shape, name = 'feat_map4')

                    feature_maps_out = tf.concat((feat_map2, feat_map4),axis=3,name='maps_out')

                    tf.add_to_collection(end_points_collection,feature_maps_out)

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return end_points

import tensorflow as tf

class BevFeatureExtractor:

    def __init__(self,extractor_config):
        self.config = extractor_config

    def preprocess_input(self, tensor_in, output_size):
        """Preprocesses the given input.
        Args:
            tensor_in: A `Tensor` of shape=(batch_size, height,
                width, channel) representing an input image.
            output_size: The size of the input (H x W)
        Returns:
            Preprocessed tensor input, resized to the output_size
        """

        tensor_out = tf.image.resize_images(tensor_in, output_size)
        
        return tensor_out
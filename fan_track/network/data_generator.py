import numpy as np
from process_data import InputData
from getKittiSeq import trackingData
import os
import tensorflow as tf
from sklearn.utils import shuffle
from fan_track.config.config import GlobalConfig


class DataGenerator:
    def __init__(self, args):
        self.filename = os.path.join(GlobalConfig.KITTI_ROOT,"secondnet/tf_record",
                                     "train.tfrecords")
        self.filename_val = os.path.join(GlobalConfig.KITTI_ROOT,"secondnet/tf_record"
                                         "val.tfrecords")
        self.args = args

    def make_batch(self, train, filename):
        # Args:
        # train: True if training/validation, False if testing
        batch_size = self.args.batch_size

        dataset = tf.data.TFRecordDataset(filenames=filename)
        # Parse the serialized data in the TFRecords files.
        # This returns TensorFlow tensors for X1,X2,Y,Z,W.
        dataset = dataset.map(parse)

        if train:
            # If training then read a buffer of the given size and randomly shuffle it.
            dataset = dataset.shuffle(buffer_size=4096)
            # Allow infinite reading of the data.
            num_repeat = None
        else:
            # If testing then don't shuffle the data.
            # Only go through the data once.
            num_repeat = 1

            # Repeat the dataset the given number of times.
        dataset = dataset.repeat(num_repeat)

        # Get a batch of data with the given size.
        dataset = dataset.batch(batch_size)

        # Create an iterator for the dataset
        iterator = dataset.make_one_shot_iterator()

        # Get the next batch of images and labels.
        x_batch, y_batch, z_batch = iterator.get_next()

        x_batch = tf.reshape(x_batch, [-1, self.args.map_length, self.args.map_width, self.args.max_targets + 1])
        y_batch = tf.reshape(y_batch, [-1, self.args.max_targets, self.args.max_meas+1])

        return x_batch, y_batch, z_batch

    def make_val_batch(self):

        # Args:
        batch_size = self.args.batch_size

        dataset = tf.data.TFRecordDataset(filenames=self.filename_val)
        # Parse the serialized data in the TFRecords files.
        # This returns TensorFlow tensors for X,Y,Z.
        dataset = dataset.map(parse)

        dataset = dataset.shuffle(buffer_size=4096)
        num_repeat = None

        # Repeat the dataset the given number of times.
        dataset = dataset.repeat(num_repeat)

        # Get a batch of data with the given size.
        dataset = dataset.batch(batch_size)

        # Create an iterator for the dataset
        iterator = dataset.make_one_shot_iterator()

        # Get the next batch of images and labels.
        x_batch, y_batch, z_batch = iterator.get_next()

        x_batch = tf.reshape(x_batch, [-1, self.args.map_length, self.args.map_width, self.args.max_targets + 1])
        y_batch = tf.reshape(y_batch, [-1, self.args.max_targets, self.args.max_meas + 1])

        return x_batch, y_batch, z_batch

    def make_test_data(self, out_path):

        # Args:
        batch_size = 1

        dataset = tf.data.TFRecordDataset(filenames=out_path)
        # Parse the serialized data in the TFRecords files.
        # This returns TensorFlow tensors for X,Y,Z.
        dataset = dataset.map(parse)

        num_repeat = 1

        # Repeat the dataset the given number of times.
        dataset = dataset.repeat(num_repeat)

        # Get a batch of data with the given size.
        dataset = dataset.batch(batch_size)

        # Create an iterator for the dataset
        iterator = dataset.make_one_shot_iterator()

        # Get the next batch of images and labels.
        x_batch, y_batch, z_batch, w_batch = iterator.get_next()

        x_batch = tf.reshape(x_batch, [-1, self.args.crop_size + 1, self.args.crop_size + 1, self.args.max_target])
        y_batch = tf.reshape(y_batch, [-1, self.args.crop_size + 1, self.args.crop_size + 1, self.args.max_target])

        return x_batch, y_batch, z_batch, w_batch


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def save_tf_record(out_path1, out_path2, args):
    # Args:
    # out_path1: File-path for the TFRecords training output file.
    # out_path2: File-path for the TFRecords validation output file.

    # Generate the tracking data
    track_loader = trackingData(gt_path= os.path.join(GlobalConfig.KITTI_ROOT,"tracking_labels"))
    track_loader.loadGroundtruth()
    input_data = InputData(gt=track_loader, calib_path_in= os.path.join(GlobalConfig.KITTI_ROOT, "tracking_calib/training"), args = args)
    input_data.generation(0, 21)

    # Convert from list to array
    x_data = np.array(input_data.cropped_maps)
    y_data = np.array(input_data.labels)
    z_data = np.array(input_data.target_no)
    w_data = np.array(input_data.meas_no)

    # Data types
    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(np.int64)
    z_data = z_data.astype(np.int64)
    w_data = w_data.astype(np.int64)

    # Partition data
    x_data, y_data, z_data, w_data = shuffle(x_data, y_data, z_data, w_data)
    pivot = int(0.85 * x_data.shape[0])
    train_x, val_x = x_data[:pivot], x_data[pivot:]
    train_y, val_y = y_data[:pivot], y_data[pivot:]
    train_z, val_z = z_data[:pivot], z_data[pivot:]
    train_w, val_w = w_data[:pivot], w_data[pivot:]

    # Create TF records
    create_tf_records(out_path1, train_x, train_y, train_z, train_w)
    create_tf_records(out_path2, val_x, val_y, val_z, val_w)


def parse(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.

    features = \
        {
            'X': tf.FixedLenFeature([], tf.string),
            'Y': tf.FixedLenFeature([], tf.string),
            'Z': tf.FixedLenFeature([], tf.int64),
            'W': tf.FixedLenFeature([], tf.int64)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Decode the raw bytes so it becomes a tensor with type.
    x = tf.decode_raw(parsed_example['X'], tf.float32)
    y = tf.decode_raw(parsed_example['Y'], tf.int64)
    z = parsed_example['Z']
    w = parsed_example['W']

    return x, y, z, w


def create_tf_records(out_path, x_data, y_data, z_data, w_data):
    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
        # Iterate over all the arrays
        for (x, y, z, w) in zip(x_data, y_data, z_data, w_data):
            # Convert the image to raw bytes.
            x_bytes = x.tostring()
            y_bytes = y.tostring()

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                {
                    'X': wrap_bytes(x_bytes),
                    'Y': wrap_bytes(y_bytes),
                    'Z': wrap_int64(z),
                    'W': wrap_int64(w)

                }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)


def save_test_record(out_path):
    # Args:
    # out_path: File-path for the TFRecords testing output file.

    # Generate the tracking data
    track_loader = trackingData(gt_path="avod_car_kitti_testing/010")
    track_loader.loadGroundtruth()
    map_data = MapData(gt=track_loader, calib_path_in="Datasets/kitti_tracking/" \
                                                      "data_tracking_calib/testing/calib/")
    map_data.generation(0, 1)
    sim_arr = np.array(map_data.sim_maps)
    loc_arr = np.reshape(np.array(map_data.location_maps), [-1, sim_arr.shape[1], sim_arr.shape[2], 1])

    # Convert from list to array
    x_data = np.concatenate((loc_arr, sim_arr), axis=3)
    y_data = np.array(map_data.labels)
    z_data = np.array(map_data.target_no)
    w_data = np.array(map_data.meas_no)

    # Data types
    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(np.int64)
    z_data = z_data.astype(np.int64)
    w_data = w_data.astype(np.int64)

    # Create TF records
    create_tf_records(out_path, x_data=x_data, y_data=y_data, z_data=z_data, w_data=w_data)

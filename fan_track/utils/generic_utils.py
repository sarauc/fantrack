import numpy as np
import tensorflow as tf
import cv2
import argparse
from fan_track.config.config import *
from pathlib import Path

def get_project_root():
    """Returns project root folder."""
    return str(Path(__file__).parent.parent.resolve())

class TrainingType(object):
        Both  = 0
        Simnet  = 1
        Assocnet = 2
        SimnetRun = 3
        Inference = 4

class ObjectLabel:
    """Object Label Class
    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                      'Misc' or 'DontCare'
    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                      truncated refers to the object leaving image boundaries
    1    occluded     Integer (0,1,2,3) indicating occlusion state:
                      0 = fully visible, 1 = partly occluded
                      2 = largely occluded, 3 = unknown
    1    alpha        Observation angle of object, ranging [-pi..pi]
    4    bbox         2D bounding box of object in the image (0-based index):
                      contains left, top, right, bottom pixel coordinates
    3    dimensions   3D object dimensions: height, width, length (in meters)
    3    location     3D object location x,y,z in camera coordinates (in meters)
    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
    1    score        Only for results: Float, indicating confidence in
                      detection, needed for p/r curves, higher is better.
    """

    def __init__(self):
        self.type = ""  # Type of object
        self.truncation = 0.
        self.occlusion = 0.
        self.alpha = 0.
        self.x1 = 0.
        self.y1 = 0.
        self.x2 = 0.
        self.y2 = 0.
        self.h = 0.
        self.w = 0.
        self.l = 0.
        self.t = (0., 0., 0.)
        self.ry = 0.
        self.score = 0.

    def __eq__(self, other):
        """Compares the given object to the current ObjectLabel instance.
        :param other: object to compare to this instance against
        :return: True, if other and current instance is the same
        """
        if not isinstance(other, ObjectLabel):
            return False

        if self.__dict__ != other.__dict__:
            return False
        else:
            return True

def trim_txt_files(*directories):
    """Remove trailing whitespace on all .txt files in the given directories.
    """
    nchanged = 0
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for fname in files:
                filename = os.path.join(root, fname)
                if fname.endswith('.txt'):
                    with open(filename, 'rb') as f:
                        code1 = f.read().decode()
                    lines = [line.rstrip() for line in code1.splitlines()]
                    while lines and not lines[-1]:
                        lines.pop(-1)
                    lines.append('')  # always end with a newline
                    code2 = '\n'.join(lines)
                    if code1 != code2:
                        nchanged += 1
                        print('  Removing trailing whitespace on', filename)
                        with open(filename, 'wb') as f:
                            f.write(code2.encode())
    print('Removed trailing whitespace on {} files.'.format(nchanged))

def check_box_3d_format(input_data):

    """Checks for correct box_3d format. If not proper type, raises error.
    Args:
        input_data: input numpy array or tensor to check for valid box_3d format
    """

    # Check type
    if isinstance(input_data, np.ndarray):
        # Check for size for numpy array form (N x 7)
        if input_data.ndim == 2:
            if input_data.shape[1] != 7:
                raise TypeError('Given input does not have valid number of '
                                'attributes. Should be N x 7 for box_3d.')
        elif input_data.ndim == 1:
            if input_data.shape[0] != 7:
                raise TypeError('Given input does not have valid number of '
                                'attributes. Should be 7 for box_3d.')

    elif isinstance(input_data, tf.Tensor):
        # if tensor, check the shape
        if isinstance(input_data, tf.Tensor):
            if input_data.shape[1] != 7:
                raise TypeError('Given input does not have valid number of '
                                'attributes. Should be N x 7 for box_3d.')
    else:
        raise TypeError('Given input is not of valid types.''(i.e. np.ndarray or tf.Tensor)')

def box_3d_to_object_label(box_3d, obj_type='Car'):
    """Turns a box_3d into an ObjectLabel
    Args:
        box_3d: 3D box in the format [x, y, z, l, w, h, ry]
        obj_type: Optional, the object type
    Returns:
        ObjectLabel with the location, size, and rotation filled out
    """

    check_box_3d_format(box_3d)

    obj_label = ObjectLabel()

    obj_label.type = obj_type

    obj_label.t = box_3d.take((0, 1, 2))
    obj_label.l = box_3d[3]
    obj_label.w = box_3d[4]
    obj_label.h = box_3d[5]
    obj_label.ry = box_3d[6]

    return obj_label

def project_to_image(point_cloud, p):
    """ Projects a 3D point cloud to 2D points for plotting
    :param point_cloud: 3D point cloud (3, N)
    :param p: Camera matrix (3, 4)
    :return: pts_2d: the image coordinates of the 3D points in the shape (2, N)
    """

    pts_2d = np.dot(p, np.append(point_cloud,
                                 np.ones((1, point_cloud.shape[1])),
                                 axis=0))

    pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
    pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]
    pts_2d = np.delete(pts_2d, 2, 0)

    return pts_2d

def compute_box_corners_3d(object_label):
    """
    Computes the 3D bounding box corner positions from an ObjectLabel
    :param object_label: ObjectLabel to compute corners from
    :return: a numpy array of 3D corners if the box is in front of the camera,
             an empty array otherwise
    """

    # compute rotational matrix
    rot = np.array([[+np.cos(object_label.ry), 0, +np.sin(object_label.ry)],
                    [0, 1, 0],
                    [-np.sin(object_label.ry), 0, +np.cos(object_label.ry)]])

    l = object_label.l
    w = object_label.w
    h = object_label.h

    # 3D BB corners
    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))

    corners_3d[0, :] = corners_3d[0, :] + object_label.t[0]
    corners_3d[1, :] = corners_3d[1, :] + object_label.t[1]
    corners_3d[2, :] = corners_3d[2, :] + object_label.t[2]

    return corners_3d

def project_to_image_space(box_3d, calib_p2,
                           truncate=False, image_shape=None,
                           discard_before_truncation=True):

    """ Projects a box_3d into image space
    Args:
        box_3d: single box_3d to project
        calib_p2: stereo calibration p2 matrix
        truncate: if True, 2D projections are truncated to be inside the image
        image_shape: [w, h] must be provided is truncate is True,
            used for truncation
        discard_before_truncation: If True, discard boxes that are larger than
            80% of the image in width OR height BEFORE truncation. If False,
            discard boxes that are larger than 80% of the width AND
            height AFTER truncation.
    Returns:
        Projected box in image space [x1, y1, x2, y2]
            Returns None if box is not inside the image
    """

    check_box_3d_format(box_3d)

    obj_label = box_3d_to_object_label(box_3d)
    corners_3d = compute_box_corners_3d(obj_label)

    projected = project_to_image(corners_3d, calib_p2)

    x1 = np.amin(projected[0])
    y1 = np.amin(projected[1])
    x2 = np.amax(projected[0])
    y2 = np.amax(projected[1])

    img_box = np.array([x1, y1, x2, y2])

    if truncate:
        if not image_shape:
            raise ValueError('Image size must be provided')

        image_w = image_shape[0]
        image_h = image_shape[1]

        # Discard invalid boxes (outside image space)
        if img_box[0] > image_w or \
                img_box[1] > image_h or \
                img_box[2] < 0 or \
                img_box[3] < 0:
            return None

        # Discard boxes that are larger than 80% of the image width OR height
        if discard_before_truncation:
            img_box_w = img_box[2] - img_box[0]
            img_box_h = img_box[3] - img_box[1]
            if img_box_w > (image_w * 0.8) or img_box_h > (image_h * 0.8):
                return None

        # Truncate remaining boxes into image space
        if img_box[0] < 0:
            img_box[0] = 0
        if img_box[1] < 0:
            img_box[1] = 0
        if img_box[2] > image_w:
            img_box[2] = image_w
        if img_box[3] > image_h:
            img_box[3] = image_h

        # Discard boxes that are covering the the whole image after truncation
        if not discard_before_truncation:
            img_box_w = img_box[2] - img_box[0]
            img_box_h = img_box[3] - img_box[1]
            if img_box_w > (image_w * 0.8) and img_box_h > (image_h * 0.8):
                return None

    return img_box

def draw(x, image, label, frame_no):

    """
        Draw 2d bounding box on an image.
        Input: x is the 2d bounding box coordinates
               [x1,y1,x2,y2].
               image is the image where bounding box
               is drawn.
               label is the identity of the target.
    """

    # top-left and bottom-right corners
    x1 = int(x[0])
    x2 = int(x[2])

    y1 = int(x[1])
    y2 = int(x[-1])

    # draw bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), ((label*100)%255, (120*label+50)%255, (50*label+20)%255), 3)

    # annotation, i.e., label
    cv2.putText(img = image,
                text = 'Id:' + str(label),
                org = (x1+1,y1-1),
                fontFace = cv2.FONT_HERSHEY_PLAIN,
                fontScale = 1.5,
                color = (255,255,255),
                thickness = 2)

    cv2.putText(img=image,
                text='Frame:' + frame_no,
                org=(20, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 255),
                thickness=2)

def read_calib_matrix(calib_file):
    '''
        Read the camera calibration matrix.
        Input: calib_file is the calibration file.
    '''

    with open(calib_file,'r') as f:

        for line in f:

            # remove the whitespace characs at the end
            line = line.rstrip()

            # convert string into a list of strings
            info = line.split()

            if info[0] == 'P2:':

                # convert str to float reading row-aligned data
                num_val = [float(x) for x in info[1:]]

                calib_p2 = np.asarray(num_val, np.float32)

                calib_p2 = np.reshape(calib_p2,(3,4))

                break

    return calib_p2

def read_img(img_path):
    '''
        Read the current image.
        Inputs: img_path is the path to image files.
        Return: rgb image of the current scene.
    '''

    # Load an color image without changing
    frame = cv2.imread(img_path)

    # since OpenCV follows BGR order
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return rgb_img

def cam_to_imu_transform(calib_dir):
    '''
        To transform a point X from GPS/IMU coordinates to the 3D camera coordinates:
                    Y =  (R|T)_velo_to_cam * (R|T)_imu_to_velo * X,
        where
            - (R|T)_velo_to_cam (4x4): velodyne coordinates -> cam 0 coordinates
            - (R|T)_imu_to_velo (4x4): imu coordinates -> velodyne coordinates
        Inputs: calib_dir is the directory tree of the calibration file.
        Output: 4x4 transformation matrix from 3D camera coordinates to GPS/IMU
                coordinates.
    '''

    row4 = np.asarray([0, 0, 0, 1], dtype=np.float32)

    with open(calib_dir, 'r') as f:

        for line in f:

            # remove the whitespace characs at the end
            line = line.rstrip()

            # convert string into a list of strings
            info = line.split()

            if (info[0] == 'Tr_velo_cam'):

                # convert str to float reading row-aligned data
                num_val = [float(x) for x in info[1:]]

                Tr_velo_cam = np.asarray(num_val, np.float32)

                # transformation matrix  in homogenous coordinates
                Tr_velo_cam = np.vstack((Tr_velo_cam.reshape((3, 4)), row4))

            elif (info[0] == 'Tr_imu_velo'):

                # convert str to float reading row-aligned data
                num_vals = [float(x) for x in info[1:]]

                Tr_imu_velo = np.asarray(num_vals, np.float32)

                # transformation matrix  in homogeneous coordinates
                Tr_imu_velo = np.vstack((Tr_imu_velo.reshape((3, 4)), row4))

                # transform a point X from GPS/IMU coordinates to the camera coordinates
    Tr_imu_cam = np.dot(Tr_imu_velo, Tr_velo_cam)

    # return cam to imu/gps transformation matrix
    return np.linalg.inv(Tr_imu_cam)

def prepare_args():

    parser = argparse.ArgumentParser()

    # map width
    parser.add_argument('--map_width', type=int, default=TrackerConfig.MAP_WIDTH,
                        help='width of map')
    # map length
    parser.add_argument('--map_length', type=int, default=TrackerConfig.MAP_LENGTH,
                        help='length of map')
    # max targets
    parser.add_argument('--max_targets', type=int, default=TrackerConfig.MAX_TARGETS,
                        help='max possible targets in one frame')

    # size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size')
    # keep checkpoints
    parser.add_argument('--max_to_keep', type=int, default=0,
                        help='the maximum number of recent checkpoint files to keep.')

    # Training Type
    parser.add_argument('--training_type', type=int, default = TrainingType.Assocnet,
                         help='0 - Train both the networks; 1 - Train the Simnet; 2 - Train the Assocnet')

    parser.add_argument('--epochs', type = float, default = 100,
                        help = 'number of training epochs')

    parser.add_argument('--disp_rate', type = float, default = 2,
                        help = 'Display Rate during training')

    parser.add_argument('--save_path', type = str, default = os.path.join(get_project_root(), 'data'),
                        help = 'Path for saving checkpoints')

    parser.add_argument('--save_rate', type = int, default = 5,
                        help = ' save network weights in training at this rate')

    parser.add_argument('--simnet_skewness', type = float, default = 1.4,
                        help = 'weights of negatives in the loss because of skewness')

    parser.add_argument('--avod_checkpoint_name', type = str, default = TrackerConfig.AVOD_CKPT_NAME,
                        help = 'Name of the avod checkpoint to use')

    parser.add_argument('--skewness_undetection', type = float, default = 0.8,
                        help = 'Skewness weight for undetected targets in the loss function')

    parser.add_argument('--simnet_ckpt_path', type = str, default = SimnetConfig.SIMNET_CKPT_PATH,
     help = 'Location for Simnet checkpoint')

    parser.add_argument('--assocnet_ckpt_path', type=str, default = AssocnetConfig.ASSOCNET_CKPT_PATH,
     help='Location for Assocnet checkpoint')

    args = parser.parse_args()
    args.training_type = TrainingType.SimnetRun

    return args

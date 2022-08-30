import csv
import glob
import os
from collections import namedtuple
import numpy as np
import pathlib2
from avod.core import box_3d_encoder
from avod.core import box_3d_projector
from pykitti import utils as kitti_utils
from scipy import stats
from wavedata.tools.core import calib_utils as calib
from wavedata.tools.core.calib_utils import FrameCalibrationData

def read_calibration(calib_dir, seq_idx):
    """Reads in Calibration file from Kitti Dataset.

    Keyword Arguments:
    ------------------;
    calib_dir : Str
                Directory of the calibration files.

    img_idx : Int
              Index of the image.

    cam : Int
          Camera used from 0-3.

    Returns:
    --------
    frame_calibration_info : FrameCalibrationData
                             Contains a frame's full calibration data.

    """
    frame_calibration_info = FrameCalibrationData()

    data_file = open(calib_dir + "/%04d.txt" % seq_idx, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:13]
        p = [float(x) for x in p]
        p = np.reshape(p, (3, 4))
        p_all.append(p)

    frame_calibration_info.p0 = p_all[0]
    frame_calibration_info.p1 = p_all[1]
    frame_calibration_info.p2 = p_all[2]
    frame_calibration_info.p3 = p_all[3]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = [float(x) for x in tr_rect[1:10]]
    frame_calibration_info.r0_rect = np.reshape(tr_rect, (3, 3))

    bot = np.array([[0, 0, 0, 1]])
    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = [float(x) for x in tr_v2c[1:13]]
    frame_calibration_info.tr_velodyne_to_cam = np.reshape(tr_v2c, (3, 4))

    # Read in imu to velodyne matrix
    tr_i2v = data[6]
    tr_i2v = [float(x) for x in tr_i2v[1:13]]
    frame_calibration_info.tr_imu_to_velodyne = np.reshape(tr_i2v, (3, 4))

    return frame_calibration_info

def get_lidar_point_cloud(img_idx, velo_dir,frame_calib,
                          im_size=None, min_intensity=None):
    """ Calculates the lidar point cloud, and optionally returns only the
    points that are projected to the image.

    :param img_idx: image index
    :param calib_dir: directory with calibration files
    :param velo_dir: directory with velodyne files
    :param im_size: (optional) 2 x 1 list containing the size of the image
                      to filter the point cloud [w, h]
    :param min_intensity: (optional) minimum intensity required to keep a point

    :return: (3, N) point_cloud in the form [[x,...][y,...][z,...]]
    """

    # Read calibration info

    x, y, z, i = calib.read_lidar(velo_dir=velo_dir,
                                  img_idx=img_idx)

    # Calculate the point cloud
    pts = np.vstack((x, y, z)).T
    pts = calib.lidar_to_cam_frame(pts, frame_calib)

    # The given image is assumed to be a 2D image
    if not im_size:
        point_cloud = pts.T
        return point_cloud

    else:
        # Only keep points in front of camera (positive z)
        pts = pts[pts[:, 2] > 0]
        point_cloud = pts.T

        # Project to image frame
        point_in_im = calib.project_to_image(point_cloud, p=frame_calib.p2).T

        # Filter based on the given image size
        image_filter = (point_in_im[:, 0] > 0) & \
                       (point_in_im[:, 0] < im_size[0]) & \
                       (point_in_im[:, 1] > 0) & \
                       (point_in_im[:, 1] < im_size[1])

    if not min_intensity:
        return pts[image_filter].T

    else:
        intensity_filter = i > min_intensity
        point_filter = np.logical_and(image_filter, intensity_filter)
        return pts[point_filter].T

""" The following functions are modified from the kitti raw data utility
https://github.com/pratikac/kitti
"""
def poses_from_oxts(oxts_packets):
    """Helper method to compute SE(3) pose matrices from OXTS packets."""
    er = 6378137.  # earth radius (approx.) in meters

    # compute scale from first lat value
    scale = np.cos(oxts_packets[0].lat * np.pi / 180.)

    t_0 = []    # initial position
    poses = []  # list of poses computed from oxts
    for packet in oxts_packets:
        # Use a Mercator projection to get the translation vector
        tx = scale * packet.lon * np.pi * er / 180.
        ty = scale * er * \
            np.log(np.tan((90. + packet.lat) * np.pi / 360.))
        tz = packet.alt
        t = np.array([tx, ty, tz])

        # We want the initial position to be the origin, but keep the ENU
        # coordinate system
        if len(t_0) == 0:
            t_0 = t

        # Use the Euler angles to get the rotation matrix
        Rx = kitti_utils.rotx(packet.roll)
        Ry = kitti_utils.roty(packet.pitch)
        Rz = kitti_utils.rotz(packet.yaw)
        R = Rz.dot(Ry.dot(Rx))

        # Combine the translation and rotation into a homogeneous transform
        poses.append(kitti_utils.transform_from_rot_trans(R, t - t_0))

    return poses

def load_oxts(oxts_dir, frame_range = None):
    """Load OXTS data from file."""

    # Find all the data files
    oxts_path = os.path.join(oxts_dir, '*.txt')
    oxts_files = sorted(glob.glob(oxts_path))

    # Subselect the chosen range of frames, if any
    if frame_range:
        oxts_files = [oxts_files[i] for i in self.frame_range]

    print('Found ' + str(len(oxts_files)) + ' OXTS measurements...')

    # Extract the data from each OXTS packet
    # Per dataformat.txt
    OxtsPacket = namedtuple('OxtsPacket',
                            'lat, lon, alt, ' +
                            'roll, pitch, yaw, ' +
                            'vn, ve, vf, vl, vu, ' +
                            'ax, ay, az, af, al, au, ' +
                            'wx, wy, wz, wf, wl, wu, ' +
                            'pos_accuracy, vel_accuracy, ' +
                            'navstat, numsats, ' +
                            'posmode, velmode, orimode')

    
    # Bundle into an easy-to-access structure
    OxtsData = namedtuple('OxtsData', 'packet, tr_imu_to_world')
    oxts = []
    for filename in oxts_files:
        oxts_packets = []
        oxts_seq = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()
                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                data = OxtsPacket(*line)
                oxts_packets.append(data)
        # Precompute the IMU poses in the world frame
        tr_imu_to_world = poses_from_oxts(oxts_packets)
        for (p, T) in zip(oxts_packets, tr_imu_to_world):
            oxts_seq.append(OxtsData(p, T))
        oxts.append(oxts_seq)

    print('done.')
    return(oxts)

def get_tr_cam_to_world(calib_dir, oxts_dir):
    """ Calculates the transform from camera_F coordinates to world coordinates for each sequence and frames

    :param calib_dir: calibration directory
    :param oxts_dir: oxts directory

    :return: list of transforms 
    """
    oxts = load_oxts(oxts_dir)
    bot = np.array([[0, 0, 0, 1]])
    tr_cam_to_world = []
    for seq_idx in range(len(oxts)):
        frame_calib = read_calibration(calib_dir=calib_dir, seq_idx=seq_idx)
        tr_cam_to_world_seq = []
        oxts_seq = oxts[seq_idx]
        for oxts_f in oxts_seq:
            tr_velodyne_to_cam = np.concatenate((frame_calib.tr_velodyne_to_cam, bot), axis=0)
            tr_imu_to_velodyne = np.concatenate((frame_calib.tr_imu_to_velodyne, bot), axis=0)
            tr_cam_to_imu = np.linalg.inv(np.matmul(tr_velodyne_to_cam,tr_imu_to_velodyne))
            
            tr_imu_to_world = oxts_f.tr_imu_to_world
            tr_cam_to_world_f = np.matmul(tr_imu_to_world,tr_cam_to_imu)
            tr_cam_to_world_seq.append(tr_cam_to_world_f)
        tr_cam_to_world.append(tr_cam_to_world_seq)

    return tr_cam_to_world

def load_detection(predictions_path, score_threshold=0.0):
    """
           
           Determine significant predictions using the prediction threshold.
           returns: 3D bboxes in the format of [x, y, z, l, w, h, ry], their scores and class indices.
    """
    # Load predictions from file
    predictions_and_scores = np.loadtxt(predictions_path)
                                                                 
    prediction_boxes_3d = predictions_and_scores[:, 0:7]
    prediction_scores = predictions_and_scores[:, 7]
    prediction_class_indices = predictions_and_scores[:, 8]
    
    if (len(prediction_boxes_3d) > 0):

        # return significant predictions 
        score_mask = prediction_scores >= score_threshold
        prediction_boxes_3d = prediction_boxes_3d[score_mask]
        prediction_scores = prediction_scores[score_mask]
        prediction_class_indices = prediction_class_indices[score_mask]
    
    return prediction_boxes_3d,prediction_scores,prediction_class_indices

def write_tracking_data(track_path, tracks):
    pathlib2.Path(track_path).mkdir(parents=True, exist_ok=True) 
    for seq_idx, track_seq in enumerate(tracks):
        track_dir = track_path+'/%04d.txt' %seq_idx
        with open(track_dir,"w") as f:
            for track_f in track_seq:
                for track in track_f:
                    """
                    Save it in Kitti tracking data format
                        frame        Frame within the sequence where the object appearers
                        track id     Unique tracking id of this object within this sequence
                        type         Describes the type of object: 'Car', 'Van', 'Truck',
                                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                                     'Misc' or 'DontCare'
                        truncated    Integer (0,1,2) indicating the level of truncation.
                                     Note that this is in contrast to the object detection
                                     benchmark where truncation is a float in [0,1].
                        occluded     Integer (0,1,2,3) indicating occlusion state:
                                     0 = fully visible, 1 = partly occluded
                                     2 = largely occluded, 3 = unknown
                        alpha        Observation angle of object, ranging [-pi..pi]
                        bbox         2D bounding box of object in the image (0-based index):
                                     contains left, top, right, bottom pixel coordinates
                        dimensions   3D object dimensions: height, width, length (in meters)
                        location     3D object location x,y,z in camera coordinates (in meters)
                        rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
                        score        Only for results: Float, indicating confidence in
                                     detection, needed for p/r curves, higher is better.
                    """
                    out_str = str(track.frame) + ' ' + str(track.track_id) + ' ' +  track.obj_type.lower() + ' ' + \
                              str(track.truncation) + ' ' + str(track.occlusion) + ' ' + str(track.obs_angle) + ' ' + \
                              str(track.x1) + ' ' + str(track.y1) + ' ' + str(track.x2) + ' ' + str(track.y2) + ' ' + \
                              str(track.h) + ' ' + str(track.w) + ' ' + str(track.l) + ' ' + str(track.X) + ' ' + str(track.Y) + ' ' + \
                              str(track.Z) + ' ' + str(track.yaw) + ' ' + str(track.score) + '\n'
                    f.write(out_str)

def boxoverlap(a,b,criterion="union"):
    """
        boxoverlap computes intersection over union for bbox a and b in KITTI format.
        If the criterion is 'union', overlap = (a inter b) / a union b).
        If the criterion is 'a', overlap = (a inter b) / a, where b should be a dontcare area.
    """
    
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    w = x2-x1
    h = y2-y1

    if w<=0. or h<=0.:
        return 0.
    inter = w*h
    aarea = (a.x2-a.x1) * (a.y2-a.y1)
    barea = (b.x2-b.x1) * (b.y2-b.y1)
    # intersection over union overlap
    if criterion.lower()=="union":
        o = inter / float(aarea+barea-inter)
    elif criterion.lower()=="a":
        o = float(inter) / float(aarea)
    else:
        raise TypeError("Unkown type for criterion")
    return o

def occ_gaussian_KDE(pts, width, height,width_resolution,height_resolution):
    """
       pts are point clound in camera frame ( y axis point down)
       the occ grid will on x: [-width/2:width/2], z: [0:height]
    """
    X, Z = np.mgrid[-width/2:width/2:width_resolution*1j,0:height:height_resolution*1j]
    positions = np.vstack([X.ravel(), Z.ravel()])
    values = np.vstack([pts[:,0], pts[:,2]])
    kde = stats.gaussian_kde(values)
    kde.set_bandwidth(bw_method=kde.factor / 10.)
    density = np.reshape(kde(positions).T, X.shape)

    return positions,density

if __name__ == "__main__":
    dataset = 'training'
    velo_dir = '/media/j7deng/Data/track/Kitti-dataset/tracking/data_tracking_velodyne/'+dataset+'/velodyne'
    calib_dir= '/media/j7deng/Data/track/Kitti-dataset/tracking/data_tracking_calib/'+dataset+'/calib'
    prediction_dir = '/home/j7deng/Kitti/tracking/detector_label/AVOD_car/'+dataset
    img_dir = '/home/j7deng/Kitti/tracking/data_tracking_image_2/'+dataset

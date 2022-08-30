import os
import functools
import numpy as np
import sys
import matplotlib.pyplot as plt
import fan_track.data_generation.position_shape_utils as ps_utils

def cam_to_imu_transform(calib_dir):
    '''
        This method returns transformation matrix to convert from 3D camera coordinates to GPS/IMU coordinates

        To transform a point X from GPS/IMU coordinates to the 3D camera coordinates:
                    Y =  (R|T)_velo_to_cam * (R|T)_imu_to_velo * X,
        where
            - (R|T)_velo_to_cam (4x4): velodyne coordinates -> cam 0 coordinates
            - (R|T)_imu_to_velo (4x4): imu coordinates -> velodyne coordinates

        Inputs: calib_dir is the directory tree of the calibration file.
        Output: 4x4 transformation matrix from 3D camera coordinates to GPS/IMU
                coordinates.
    '''

    # transform a point X from GPS/IMU coordinates to the camera coordinates
    Tr_imu_cam = imu_to_cam_transform(calib_dir)

    # return cam to imu/gps transformation matrix
    return np.linalg.inv(Tr_imu_cam)

def imu_to_cam_transform(calib_dir):
    '''
           To transform a point X from GPS/IMU coordinates to the 3D camera coordinates:
                       Y =  (R|T)_velo_to_cam * (R|T)_imu_to_velo * X,
           where
               - (R|T)_velo_to_cam (4x4): velodyne coordinates -> cam 0 coordinates
               - (R|T)_imu_to_velo (4x4): imu coordinates -> velodyne coordinates
           Inputs: calib_dir is the directory tree of the calibration file.
           Output: 4x4 transformation matrix from GPS/IMU coordinates to 3D camera
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

    return Tr_imu_cam


def mercator_projection(lat,lon,att,lat0):
    '''
        Project the GPS/IMU measurements to a planar map.
    '''
    r = 6378137

    s = np.cos(lat0*np.pi/180.0)

    # east position along the equator
    pos_x = s*r*np.pi*lon/180

    # north position wrt central meridian
    pos_y = s*r*np.log(np.tan(np.pi*(0.25 + lat/360)))

    # attitude is same as z
    pos_z = att

    return np.asarray([[pos_x, pos_y, pos_z]], dtype = np.float64)

def get_rot_mat(rx,ry,rz):
    '''
        Construct the 3D rotation matrix from oxts to the world map.
    '''

    # basic rotation matrices
    Rx = np.asarray([[1, 0, 0],
                     [0, np.cos(rx), -np.sin(rx)],
                     [0, np.sin(rx), np.cos(rx)]], dtype = np.float64)

    Ry = np.asarray([[np.cos(ry), 0, np.sin(ry)],
                     [0, 1, 0],
                     [-np.sin(ry), 0, np.cos(ry)]], dtype = np.float64)

    Rz = np.asarray([[np.cos(rz), -np.sin(rz), 0],
                     [np.sin(rz), np.cos(rz), 0],
                     [0, 0, 1]], dtype = np.float64)

    R  =  functools.reduce(np.dot, [Rz,Ry,Rx])

    return R

def concatenate(R,t):

    # rotation and translation matrix [R|t]
    RT = np.hstack((R,t.T))

    row4 = np.asarray([0,0,0,1], dtype = np.float32)

    # homogenous transformation matrix
    RT_h = np.vstack((RT,row4))

    return RT_h


def oxts_prev_frame_trans(path_oxts = None,video_no = 0, dataset_name = 'training'):
    '''
         Compute the pose{i} contains the transformation which takes
         a 3D point in the i'th frame and projects it into the oxts
         coordinates of the (i-1)st frame. In addition, compute the
         change in the yaw angle between those two frames.\
         Inputs: path_oxts is the path to the location of oxts folders.
                 video no is the integer used to name the directory of
                 the oxts file.
         Outputs: oxts_projs is the list of 4x4 transformation matrices.
                  delta_yaw is the list of changes in yaw angle, i.e.,
                  delta_yaw_i = yaw_i - yaw_(i-1).
    '''

    if (path_oxts is None):
        path_oxts = ps_utils.kitti_dir + \
                           '/{0}/{1}/{2}'.format('tracking_dataset','oxts', dataset_name)

    oxts_dir = path_oxts + '/{0:0>4}'.format(str(video_no)) + '.txt'

    # projection matrices to the oxts coordinates in the 1st frame
    oxts_projs = []

    # change in yaw angle of the oxts between ith and (i-1)st frames
    delta_yaw = []

    # the latitude of the first frame's coordinates
    lat_0 = None

    # transform inertial coordinates to the 1st oxts coordinates
    Rt_0 = None

    try:

        # read the imu gps data for each frame for the given video
        with open(oxts_dir,'r') as f:

            for k,line in enumerate(f):

                l = line.rsplit()

                gps_imu = [float(x) for x in l[0:6]]

                lat, lon, att = gps_imu[0:3]

                if (k==0 or lat_0 is None):
                    lat_0 = lat

                t = mercator_projection(lat,lon,att,lat_0)

                # rotations:
                rx = gps_imu[3]  # roll around the x-axis
                ry = gps_imu[4]  # pitch around the y-axis
                rz = gps_imu[5]  # heading around the z-axis

                R = get_rot_mat(rx,ry,rz)

                # ith oxts coordinates -> the world coordinates
                Rt_h = concatenate(R,t)

                # normalization matrix to start start at (0,0,0)
                if (k == 0 or Rt_0 is None):
                    Rt_0 = Rt_h
                    delta_yaw.append(0)
                    delta_yaw.append(-rz)

                # the world coordinates -> (k-1)st oxts coordinates
                oxts_proj_i = np.linalg.solve(Rt_0,Rt_h)

                # set (0,0,0) the oxts coordinates at time k>0
                if (k>0):
                    Rt_0 = Rt_h
                    delta_yaw[k] += rz
                    delta_yaw.append(-rz)

                oxts_projs.append(oxts_proj_i)

    except IOError as e:
        print("Could not read file:{0.filename}".format(e))
        sys.exit()

    return oxts_projs, delta_yaw


def oxts_0_frame_trans(path_oxts = None,video_no = 0, dataset_name = 'training'):
    '''
         Compute the pose{i} contains the transformation which takes
         a 3D point in the i'th frame and projects it into the oxts
         coordinates of the (i-1)st frame. In addition, compute the
         change in the yaw angle between those two frames.\
         Inputs: path_oxts is the path to the location of oxts folders.
                 video no is the integer used to name the directory of
                 the oxts file.
         Outputs: oxts_projs is the list of 4x4 transformation matrices.
                  delta_yaw is the list of changes in yaw angle, i.e.,
                  delta_yaw_i = yaw_i - yaw_(i-1).
    '''

    if (path_oxts is None):
        path_oxts = ps_utils.kitti_dir + \
                           '/{0}/{1}/{2}'.format('tracking_dataset','oxts', dataset_name)

    oxts_dir = path_oxts + '/{0:0>4}'.format(str(video_no)) + '.txt'

    # projection matrices to the oxts coordinates in the 0th frame
    oxts_projs = []

    # change in yaw angle of the oxts between ith and 0th frames
    # delta_yaw = []

    # the latitude of the first frame's coordinates
    lat_0 = None

    # transform inertial coordinates to the 0th oxts coordinates
    Rt_0 = None

    try:

        # read the imu gps data for each frame for the given video
        with open(oxts_dir,'r') as f:

            for k,line in enumerate(f):

                l = line.rsplit()

                gps_imu = [float(x) for x in l[0:6]]

                lat, lon, att = gps_imu[0:3]

                if (k==0 or lat_0 is None):
                    lat_0 = lat

                t = mercator_projection(lat,lon,att,lat_0)

                # rotations:
                rx = gps_imu[3]  # roll around the x-axis
                ry = gps_imu[4]  # pitch around the y-axis
                rz = gps_imu[5]  # heading around the z-axis

                R = get_rot_mat(rx,ry,rz)

                # ith oxts coordinates -> the world coordinates
                Rt_h = concatenate(R,t)

                # normalization matrix to start start at (0,0,0)
                if (k == 0 or Rt_0 is None):
                    Rt_0 = Rt_h
                    # delta_yaw[k] = 0

                # if k>0:
                #     delta_yaw[k] -= delta_yaw[0]

                # the world coordinates -> 0th oxts coordinates
                oxts_proj_i = np.linalg.solve(Rt_0,Rt_h)

                oxts_projs.append(oxts_proj_i)

    except IOError as e:
        print("Could not read file:{0.filename}".format(e))
        sys.exit()

    return oxts_projs, np.linalg.inv(oxts_projs)


if __name__ == '__main__':
    '''
        Unit testing of the implemented functions above.
        The figure must show the path of the car on the
        first oxts coordinates. The validation can be done
        by comparing the figure with that obtained from the
        run_demoVehiclePath.m in the devkit_raw_data.
    '''

    fig = plt.figure()

    ax = fig.add_subplot(111)

    path_oxts = os.path.expanduser('~/Kitti/tracking_dataset/oxts/training')

    video_no = 0

    oxts_dir = path_oxts + '/{0:0>4}'.format(str(video_no)) + '.txt'

    path_calib =  os.path.expanduser('~/Kitti/tracking_dataset/calib/training')

    calib_dir = path_calib + '/{0:0>4}'.format(str(video_no)) + '.txt'

    # IMU's position in its homogeneous coordinate system
    x = np.asarray([0,0,0,1], dtype = np.float32)

    # transformation matrices from ith oxts coord to (i-1)st oxts coord
    oxts_projs,delta_yaw = oxts_prev_frame_trans()

    # xy position wrt the first oxts coordinates
    pos0_xy = np.zeros(shape=2, dtype = np.float64)

    # compute x-y components of displacements on previous (i-1)st frame
    def back_rotation(yaw, x):

        R = np.asanyarray([[np.cos(sum(yaw)), -np.sin(sum(yaw))],
                           [np.sin(sum(yaw)),  np.cos(sum(yaw))]], dtype=np.float64)


        return np.dot(R,x)

    # read the imu gps data for each frame
    with open(oxts_dir,'r') as f:

        for k,line in enumerate(f):

            l = line.rsplit()

            gps_imu = [float(x) for x in l[0:6]]

            if (k==0):
                lat_0 = gps_imu[0]

                curr_yaw = gps_imu[5]

                # intial position at (0,0,0)
                prev_xyz = x.copy()

            # assume that the initial position is at (0,0,0)
            else:
                lat, lon, att = gps_imu[0:3]

                curr_xyz =  mercator_projection(lat,lon,att,lat_0)

                # rewrite current position in homogenous coordinates
                curr_xyz  = np.hstack((curr_xyz,[[1]]))[0]

                # the relative position at k wrt imu-gps coordinates at k-1
                curr_xyz = np.dot(oxts_projs[k],x)

                # compute displacement on the first oxts coordinate
                delta_xy = back_rotation(delta_yaw[:k], curr_xyz[:-2])

                # compute the x-y position on the first oxts coordinate
                pos0_xy += delta_xy

                ax.scatter(pos0_xy[0],pos0_xy[1], s = 10, edgecolor = 'r', facecolors='none',  marker = '*')

                prev_yaw = curr_yaw

                curr_yaw = gps_imu[5]

                prev_xyz[:] = curr_xyz[:]

        plt.show()

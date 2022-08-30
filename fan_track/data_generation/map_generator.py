import numpy as np
import random as rand
from fan_track.data_generation.kitti_simnet_dataset import get_filenames
from fan_track.data_generation.kitti_simnet_dataset import object_types
from fan_track.data_generation.kitti_imu_to_cam import cam_to_imu_transform
import fan_track.data_generation.position_shape_utils as ps_utils
import matplotlib.pyplot as plt
from collections import OrderedDict as oDict
from fan_track.utils.generic_utils import read_calib_matrix, read_img, draw, project_to_image_space
from matplotlib import colors
import cv2

class MapGenerator(object):

    def __init__(self, height=80, width=80, resolution=0.5):
        '''
            Create 2D global map of targets and 1D location
            indicator of measurements together with their
            normalized coordinates of local maps. The map
            locates targets with unique colors in the x-y
            imu-gps coordinates. The location indicator is
            the reshaped 2D binary map of measurement loca-
            tions in the x-y gps-imu coordinates.
            Inputs: height is the height of the map in meter
                    along the x-axis.
                    width is the width of the map in meter
                    along the y-axis.
                    max_rear_dist_x is the distance in the x-axis
                    behind the imu/gps in meter.
                    resolution is the bin size along the x-y axes.
        '''

        # resolution of target map in grid/meter
        self.resolution = resolution

        # number of pixels along the x axis
        self.num_pix_x = int(round(height/self.resolution))

        # number of pixels along the y axis
        self.num_pix_y = int(round(width/self.resolution))

        # maximum distance from the imu-gps
        self.max_y = self.resolution*self.num_pix_y/2

        # size of local_map centered around each target
        self.local_h = 21

        self.local_w = 21

        # global target map
        self.target_map = np.zeros((self.num_pix_x, self.num_pix_y),dtype=np.uint8)

        # location indicator
        self.meas_loc = np.zeros((self.num_pix_x, self.num_pix_y),dtype=np.int32)

    def get_size(self):
        '''
            Return the number of pixels along the x-y axes of the target map
        '''
        return self.target_map.shape

    def mapping(self,bbox_3d,num_obj):
        '''
            Compute x-y locations of objects in a global map and the normalized
            coordinates of a local map centered around each of these objects.
            Inputs: bbox_3d is the list of 3d bounding box centers, i.e.,[x, y, z]
            Outputs: two iterators. the first iterator is a list of four-tubles.
                     Each tuble is the normalized coordinates of a local map. The
                     second one is the list of two-tubles each of which is the
                     locations of objects in a global map.
        '''

        # halves of the local map dimensions
        half_w = (self.local_w-1)/2
        half_h = (self.local_h-1)/2

        # x-y indices of objects in the map
        x_idx = []
        y_idx = []

        # the list of objects that can be mapped
        self.mapped_objects = [False]*num_obj

        for i in range(num_obj):

            # compute x-y indices
            x = int(round((bbox_3d[i,0] + self.max_y) /self.resolution))
            y = int(round((bbox_3d[i,1] + self.max_y) /self.resolution))
            # y = int(round(bbox_3d[i,1] /self.resolution))

            # check if object are inside the map
            if  (x < self.num_pix_x and y < self.num_pix_y and x > 0 and y > 0):
                x_idx.append(x)
                y_idx.append(y)

                self.mapped_objects[i] = True

        # normalized top-left coordinates
        x1 = list(map(lambda x: (x - half_w)/(self.num_pix_y -1), x_idx))
        y1 = list(map(lambda y: (y - half_h)/(self.num_pix_x -1), y_idx))

        # normalized bottom-right coordinates
        x2 = list(map(lambda x: (x + half_w)/(self.num_pix_y -1), x_idx))
        y2 = list(map(lambda y: (y + half_h)/(self.num_pix_x -1), y_idx))

        return zip(x1,y1,x2,y2), zip(x_idx,y_idx)

    def map_targets(self,bbox_3d):
        '''
            Map targets and compute their normalized coordinates
            of local maps.
            Input: bbox_3d is the 3D bounding box centers given
                   by [x,y,z] in the imu-gps coordinates.
        '''

        num_tar = bbox_3d.shape[0]

        if (num_tar>0):

            _ , centers = self.mapping(bbox_3d,num_tar)

            centers = list(centers)

            # target centers used to construct local maps
            self.t_centers = np.asarray(centers,dtype=np.float32)

            num_mapped_targets = len(centers)

        else:
            self.mapped_objects = []

    def loc_ind_coords(self,bbox_3d):
        '''
            Compute x-y location indices of measurements together
            with the normalized coordinates of their local maps.
            The indices are mapped into a location indicator. This
            1d array will be used as an iterator to map measurement
            features into a global map in the simnet.
            Input: bbox_3d is the array of 3D bounding box centers
                   given by [x,y,z] in the gps_imu coordinates.
        '''
        # clear the localization map
        self.meas_loc.fill(0)

        # reshape it to a map
        self.meas_loc = self.meas_loc.reshape((self.num_pix_x, self.num_pix_y))

        num_meas = bbox_3d.shape[0]

        if (num_meas>0):
            #  compute normalized coordinates and centers
            _ ,centers = self.mapping(bbox_3d,num_meas)

            centers = list(centers)

            # measurement centers used to construct local maps
            self.m_centers = np.asarray(centers,dtype=np.float32)

            # clear all locations
            self.meas_loc.fill(-1)

            # check if there are measurements inside the map
            if (len(centers)>0):
                i,j = zip(*centers)

                # locate measurement indices
                self.meas_loc[i,j] = np.arange(len(centers))

        else:
            self.mapped_objects = []


        # reshape the locator to be unpacked along its first dimension
        self.meas_loc = self.meas_loc.reshape(-1)

    def visualize(self,ax,video_path,video_no,frame,calib_file,targets,mapped_targets,img_flag=True):

        if (img_flag):

            # read image
            img_path = video_path + '/' + video_no
            frame = '{0:0>6}'.format(frame) + '.png'
            img = read_img(img_path, frame)

            # read calibration files
            calib_p2 = read_calib_matrix(calib_file)

            # color index
            c_idx = 0

            for i,t in enumerate(targets):

                if (mapped_targets[i]):
                    # projected box in image sself.colorspace [x1, y1, x2, y2]
                    x = project_to_image_space(t,calib_p2)

                    try:
                        draw(x,img,i,self.colors[c_idx])
                        c_idx += 1

                    except IndexError:
                        print('index error')


            ax[0].imshow(img)
            plt.sca(ax[0])
            plt.title('current frame')
            ax[0].axis('off')

        # make every nth tick's label visible
        rate_vis = 5
        ax[1].imshow(self.target_map)
        plt.sca(ax[1])

        plt.yticks(np.arange(0,int(self.num_pix_x ),1),
                   np.arange(int(self.num_pix_x*self.resolution),0,-self.resolution))
        plt.setp(ax[1].get_yticklabels(), visible=False)
        plt.setp(ax[1].get_yticklabels()[::rate_vis], visible=True)
        plt.ylabel('(m)')

        plt.xticks(np.arange(0,int(self.num_pix_y ),1),
                   np.arange(int(self.num_pix_y *self.resolution/2),
                             -int(self.num_pix_y *self.resolution/2),-self.resolution))
        plt.setp(ax[1].get_xticklabels(), visible=False)
        plt.setp(ax[1].get_xticklabels()[::rate_vis], visible=True)
        plt.xlabel('(m)')

        plt.grid(True,color='white',linewidth=0.2)
        plt.title('target map (resolution {0:0.2f} m)'.format(self.resolution))

        plt.show(False)

        plt.pause(0.2)

    def save_meas_centers(self, filename):
        try:
            data = np.full((160,160,3),255)
            for m_center in self.m_centers:
                color = np.random.random_integers(10,255,3)
                i = int(m_center[0])
                j = int(m_center[1])
                data[i,j,:] = color

            cv2.imwrite(filename+'.jpg',data)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)

if __name__ == '__main__':

    axes = []

    fig_img = plt.figure(figsize=(7,7),dpi=100)
    axes.append(fig_img.add_subplot(111))

    fig_map = plt.figure(figsize=(7,7),dpi=100)
    axes.append(fig_map.add_subplot(111))

    # video no
    video_no = '0000'

    # the directory where frames exist
    video_path,_ = get_filenames('tracking_image_2','training')


    # obtain the path to the label folder
    label_path, _ = get_filenames('tracking_labels','label_02')

    calib_path,_ = get_filenames('tracking_calib','training')

    calib_file = calib_path + '/' + video_no + '.txt'

    T_cam_to_imu = cam_to_imu_transform(calib_file)

    # the current processing frame
    frame_id = None

    # targets and measurement matrices
    targets_imu =  np.empty((0,7))
    measurements_imu = np.empty((0,7))

    targets_cam = np.empty((0,7))
    measurements_cam = np.empty((0,7))

    maploc = MapGenerator()

    with open(label_path + '/' + video_no + '.txt','r') as f:

        # read each line of label file
        for line_no, line in enumerate(f):

            # split the string on whitespace to obtain a list of columns
            obj_info = line.split()

            if (line_no == 0 and frame_id is None):
                frame_id = int(obj_info[0])
            else:
                if (frame_id < int(obj_info[0])):

                    # check if at least two frames are already processed
                    if (frame_id > 0):

                        maploc.map_targets(targets_imu)

                        maploc.visualize(axes, video_path, video_no, frame_id-1,
                                         calib_file, targets_cam, maploc.mapped_objects)

                        maploc.loc_ind_coords(measurements_imu)

                        # save current measurements as targets and clear measurements for the next frame
                        targets_imu = measurements_imu
                        targets_cam = measurements_cam

                        # create a new measurement matrix and clear the contents of label and type lists
                        measurements_imu = np.empty((0,7))
                        measurements_cam = np.empty((0,7))

                    frame_id = int(obj_info[0])

            if (obj_info[2] in object_types):

                bbox_ry = np.asarray([obj_info[10:17]], dtype=np.float32)

                # convert Kitti's box to 3d box
                bbox_ry_cam = ps_utils.kitti_box_to_box_3d(bbox_ry)

                # convert from cam to imu coordinates
                bbox_pos_imu = np.dot(T_cam_to_imu,np.append(bbox_ry_cam[0,:3], 1))

                bbox_ry_imu = bbox_ry_cam.copy()

                bbox_ry_imu[0,:3] = bbox_pos_imu[:-1]

                if (frame_id == 0):
                    # add bbox params and rotation_y parameters of the target
                    targets_imu = np.append(targets_imu, bbox_ry_imu, axis=0)

                    targets_cam = np.append(targets_cam, bbox_ry_cam, axis=0)
                else:
                    measurements_imu = np.append(measurements_imu, bbox_ry_imu, axis=0)

                    measurements_cam = np.append(measurements_cam, bbox_ry_cam, axis=0)

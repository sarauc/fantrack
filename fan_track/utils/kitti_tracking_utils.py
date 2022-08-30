import os
import shutil
import cv2
import sys
from fan_track.config.config import GlobalConfig

# global scope
Kitti_dir = GlobalConfig.KITTI_DIR

num_frames = None

frames = None

if sys.version_info > (3, 0):
    python3=True
else:
    python3=False

def create_planes(data_split_dir = 'testing'):
    '''create the ground plane as y = 0*sx + 0*sy + h where sx, sz are slopes and h is the height of the sensor'''

    global num_frames
    global frames

    if python3:
        _, _, frames = next(os.walk(Kitti_dir + '/'+  data_split_dir +  '/velodyne'))
    else:
        _, _, frames = os.walk(Kitti_dir + '/'+  data_split_dir +  '/velodyne').next()

    num_frames = len(frames)

    planes_dir = Kitti_dir + '/testing/planes'

    # ground plane in the format of [ground_normal,height of the sensor] where normal is facing up
    ground_plane = [0, -1, 0, 1.73]

    for file_num in range(num_frames):

        filename = '{:06}'.format(file_num) + '.txt'

        with open(planes_dir + '/' + filename, 'w+') as f:

            for i in range(4):
                if (i==0):
                    f.write('# Matrix\n')
                elif(i==1):
                    f.write('WIDTH 4\n')
                elif(i==2):
                    f.write('HEIGHT 1\n')
                else:
                    f.write(''.join('{:.6e}'.format(elem) + ' ' for elem in ground_plane))


def duplicate_calib(data_split_dir = 'testing'):
    '''duplicate the calibration file for each frame for avod'''

    global num_frames
    global frames

    if python3:
        _, _, frames = next(os.walk(Kitti_dir + '/'+  data_split_dir +  '/velodyne'))
    else:
        _, _, frames = os.walk(Kitti_dir + '/' + data_split_dir + '/velodyne').next()

    num_frames = len(frames)


    calib_dir = Kitti_dir + '/' + data_split_dir  + '/calib'

    if python3:
        _, _, calibfiles = next(os.walk(calib_dir))
    else:
        _, _, calibfiles = os.walk(calib_dir).next()

    for file_num in range(num_frames):

        filename = '{:06}'.format(file_num) + '.txt'

        try:
            shutil.copy2(calib_dir + '/' + calibfiles[0], calib_dir + '/' + filename)

        except shutil.Error:

            continue

def rewrite_data_split_file(filename = 'test.txt'):

    split__file_dir = Kitti_dir + '/' + filename

    with  open(split__file_dir,'w') as f:

        for file_num in range(num_frames):
            f.write('{:06d}'.format(file_num) + '\n')


def create_video(pred_out_dir):
    if python3:
        _, _, frames = next(os.walk(pred_out_dir))
    else:
        _, _, frames = os.walk(pred_out_dir).next()

    frames.sort()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    f = frames[0]

    img = cv2.imread(pred_out_dir + '/' + f)

    rows, cols, _ = img.shape

    video = cv2.VideoWriter('output.avi',fourcc, 20.0, (cols,rows))

    video.write(img)

    for f in frames[1:]:
        img = cv2.imread(pred_out_dir + '/' + f)

        video.write(img)


    video.release()


if __name__ == '__main__':

    create_planes()

    duplicate_calib()

    rewrite_data_split_file()

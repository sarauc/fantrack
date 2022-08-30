import os

import numpy as np

import avod
from avod.core import anchor_projector
from avod.core import box_3d_encoder
from avod.builders.dataset_builder import DatasetBuilder
import avod.builders.config_builder_util as config_builder
from fan_track.config.config import GlobalConfig

kitti_dir = GlobalConfig.KITTI_ROOT

def box_3d_dimxyz(box_3d):
    '''
        Compute the dimx, dimy and dimz of a 3d bbox.
        Note that box_3d format:[x, y, z, l, w, h, ry]
        Input: a 3d bbox
        Output: an array of dimx,dimy, dimz
    '''

    cos_ry = np.abs(np.cos(box_3d[-1]))
    sin_ry = np.abs(np.sin(box_3d[-1]))

    dimx = box_3d[3] * cos_ry + box_3d[4] * sin_ry
    dimy = box_3d[5]
    dimz = box_3d[4] * cos_ry + box_3d[3] * sin_ry

    return np.asarray([dimx,dimy,dimz])


def kitti_box_to_box_3d(kitti_box):
    '''
        Turns an Kitti's box representation into an box_3d
        Inputs: kitti_box = [height, width, length, x, y, z, ry]
        Returns: 3D box in box_3d format [x, y, z, l, w, h, ry]
    '''

    box_3d = kitti_box.copy()

    # center
    box_3d[0][0] = kitti_box[0][3]
    box_3d[0][1] = kitti_box[0][4]
    box_3d[0][2] = kitti_box[0][5]
    # dimensions
    box_3d[0][3] = kitti_box[0][2]
    box_3d[0][4] = kitti_box[0][1]
    box_3d[0][5] = kitti_box[0][0]

    return box_3d


def bev_iou(bbox_gt,bbox_pos, dataset):

    # root_dir = avod.__path__[0] + '/data/outputs'
    #
    # experiment_config_path  =  root_dir + '/cars_reduced_size/cars_reduced_size.config'
    #
    # # Read the configurations
    # _, _, _, dataset_config = config_builder.get_configs_from_pipeline_file(
    #                               experiment_config_path, is_training=False)

    # Overwrite the defaults
    # dataset_config = config_builder.proto_to_obj(dataset_config)

    # Build the kitti dataset object
    # dataset = DatasetBuilder.build_kitti_dataset(dataset_config,use_defaults=False)

    # convert 3d_bbox to anchors and then obtain their 2d box corners on the bev image in the format of [x1, y1, x2, y2]
    gt_anchors =  box_3d_encoder.box_3d_to_anchor(bbox_gt)
    gt_bev_anchors, _ = \
        anchor_projector.project_to_bev(gt_anchors,dataset.kitti_utils.bev_extents)

    pos_anchor = box_3d_encoder.box_3d_to_anchor(bbox_pos)
    pos_bev_anchors, _ = \
        anchor_projector.project_to_bev(pos_anchor,dataset.kitti_utils.bev_extents)

    # determine the (x, y)-coordinates of the intersection rectangle
    xmin_max = max(gt_bev_anchors[0][0], pos_bev_anchors[0][0])
    zmin_max = max(gt_bev_anchors[0][1], pos_bev_anchors[0][1])
    xmax_min = min(gt_bev_anchors[0][2], pos_bev_anchors[0][2])
    zmax_min = min(gt_bev_anchors[0][3], pos_bev_anchors[0][3])

    # compute the area of intersection rectangle
    inter_area = max(0,(xmax_min - xmin_max + 1))*max(0,(zmax_min - zmin_max + 1))

    # compute the area of both the positive example and ground-truth rectangles
    box_pos_area = (pos_bev_anchors[0][2] - pos_bev_anchors[0][0] + 1) * \
                   (pos_bev_anchors[0][3] - pos_bev_anchors[0][1] + 1)

    box_gt_area = (gt_bev_anchors[0][2] - gt_bev_anchors[0][0] + 1) * \
                  (gt_bev_anchors[0][3] - gt_bev_anchors[0][1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(box_pos_area + box_gt_area - inter_area)

    # return the intersection over union value
    return iou

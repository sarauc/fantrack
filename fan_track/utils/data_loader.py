import os

from fan_track.evaluation.python import evaluate_tracking

from fan_track.evaluation.python.evaluate_tracking import tData, trackingEvaluation

import numpy as np

class dataLoader():

    def __init__(self, min_overlap=0.5, max_truncation = 0, min_height = 25, max_occlusion = 2,  cls="car", mapping_path="./data/tracking/evaluate_tracking.seqmap" ):
        t_sha = "tmp"
        ground_true_path="./data/tracking"
        self._track_loader = trackingEvaluation(t_sha=t_sha, gt_path=ground_true_path, min_overlap=min_overlap, 
         max_truncation = max_truncation, min_height = min_height, max_occlusion = max_occlusion, mail=None, cls=cls, mapping_path = mapping_path)
        self.sequence_name = self._track_loader.sequence_name
        self.n_frames = self._track_loader.n_frames

    def loadGTData(self):
        self._track_loader.loadGroundtruth()
        self.groundtruth = self._track_loader.groundtruth

    def loadTracker(self):
        self._track_loader.loadTracker()
        self.tracker = self._track_loader.tracker

    def setGTPath(self,gt_path):
        self._track_loader.gt_path = gt_path

    def setTrackerPath(self,t_path):
        self._track_loader.t_path = t_path


if __name__ == "__main__":

    data_loader = dataLoader()
    data_loader.setGTPath("./data/tracking/label_02/")
    data_loader.loadGTData()

    #print the first two frames of each sequences
    for seq_idx in range(len(data_loader.groundtruth)):
        seq_gt = data_loader.groundtruth[seq_idx]

        print("--------Sequence", seq_idx)
        # frame from 0 to 1
        for f_idx in range(2):
            gt = seq_gt[f_idx]
            for t_gt in gt:
                print(t_gt.frame, t_gt.track_id, t_gt.X, t_gt.Y, t_gt.Z)


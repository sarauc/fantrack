import os
import copy


class tData:
    """
        Utility class to load data.
    """

    def __init__(self, frame=-1, obj_type="unset", truncation=-1, occlusion=-1, \
                 obs_angle=-10, x1=-1, y1=-1, x2=-1, y2=-1, w=-1, h=-1, l=-1, \
                 X=-1000, Y=-1000, Z=-1000, yaw=-10, score=-1000, track_id=-1):
        """
            Constructor, initializes the object given the parameters.
        """

        # init object data
        self.frame = frame
        self.track_id = track_id
        self.obj_type = obj_type
        self.truncation = truncation
        self.occlusion = occlusion
        self.obs_angle = obs_angle
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = w
        self.h = h
        self.l = l
        self.X = X
        self.Y = Y
        self.Z = Z
        self.yaw = yaw


class trackingData(object):
    """
       Class to load tracking data

    """

    def __init__(self, gt_path="Datasets/kitti_tracking/training", cls="car"):

        # data and parameter
        self.gt_path = os.path.join(gt_path, "label_02")
        # class to evaluate, i.e. pedestrian or car
        self.cls = cls

        # get number of sequences and
        # get number of frames per sequence from test mapping
        filename_train_mapping = "./data/tracking/training.seqmap"
        # filename_train_mapping = "./data/tracking/testing.seqmap"
        self.n_frames = []
        self.sequence_name = []
        with open(filename_train_mapping, "r") as fh:
            for i, l in enumerate(fh):
                fields = l.split(" ")
                self.sequence_name.append("%04d" % int(fields[0]))
                self.n_frames.append(int(fields[3]) - int(fields[2]) + 1)
        fh.close()
        self.n_sequences = i + 1

        # this should be enough to hold all groundtruth trajectories
        # is expanded if necessary and reduced in any case
        self.gt_trajectories = [[] for x in range(self.n_sequences)]
        self.ign_trajectories = [[] for x in range(self.n_sequences)]

    def loadGroundtruth(self):
        """
            Helper function to load ground truth.
        """

        try:
            self._loadData(self.gt_path, cls=self.cls, loading_groundtruth=True)
        except IOError:
            return False
        return True

    def _loadData(self, root_dir, cls, loading_groundtruth=False):

        """
            Generic loader for ground truth and tracking data.
            Use loadGroundtruth() to load this data.
            Loads detections in KITTI format from textfiles.
        """
        # construct objectDetections object to hold detection data

        t_data = tData()

        seq_data = []
        n_trajectories = 0
        n_trajectories_seq = []

        for seq, s_name in enumerate(self.sequence_name):
            i = 0
            filename = os.path.join(root_dir, "%s.txt" % s_name)
            f = open(filename, "r")

            f_data = [[] for x in range(
                self.n_frames[seq])]  # current set has only 1059 entries, sufficient length is checked anyway
            ids = []
            n_in_seq = 0
            id_frame_cache = []

            for line in f:
                # KITTI tracking benchmark data format:
                # (frame,tracklet_id,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
                line = line.strip()
                fields = line.split(" ")
                # classes that should be loaded (ignored neighboring classes)
                if "car" in cls.lower():
                    classes = ["car", "van", "truck", "pedestrian", "cyclist", "tram"]
                elif "pedestrian" in cls.lower():
                    classes = ["pedestrian", "person_sitting"]
                else:
                    classes = [cls.lower()]
                classes += ["dontcare"]
                if not any([s for s in classes if s in fields[2].lower()]):
                    continue
                # get fields from table
                t_data.frame = int(float(fields[0]))  # frame
                t_data.track_id = int(float(fields[1]))  # id
                t_data.obj_type = fields[2].lower()  # object type [car, pedestrian, cyclist, ...]
                t_data.truncation = int(float(fields[3]))  # truncation [-1,0,1,2]
                t_data.occlusion = int(float(fields[4]))  # occlusion  [-1,0,1,2]
                t_data.obs_angle = float(fields[5])  # observation angle [rad]
                t_data.x1 = float(fields[6])  # left   [px]
                t_data.y1 = float(fields[7])  # top    [px]
                t_data.x2 = float(fields[8])  # right  [px]
                t_data.y2 = float(fields[9])  # bottom [px]
                t_data.h = float(fields[10])  # height [m]
                t_data.w = float(fields[11])  # width  [m]
                t_data.l = float(fields[12])  # length [m]
                t_data.X = float(fields[13])  # X [m]
                t_data.Y = float(fields[14])  # Y [m]
                t_data.Z = float(fields[15])  # Z [m]
                t_data.yaw = float(fields[16])  # yaw angle [rad]

                if not loading_groundtruth:
                    if len(fields) == 17:
                        t_data.score = -1
                    elif len(fields) == 18:
                        t_data.score = float(fields[17])  # detection score
                    else:
                        print("file is not in KITTI format")
                        return

                # do not consider objects marked as invalid
                if t_data.track_id is -1 and t_data.obj_type != "dontcare":
                    continue

                idx = t_data.frame
                # check if length for frame data is sufficient
                if idx >= len(f_data):
                    print("extend f_data", idx, len(f_data))
                    f_data += [[] for x in range(max(500, idx - len(f_data)))]
                try:
                    id_frame = (t_data.frame, t_data.track_id)
                    if id_frame in id_frame_cache and not loading_groundtruth:
                        print("Exiting...")
                        # continue # this allows to evaluate non-unique result files
                        return False
                    id_frame_cache.append(id_frame)
                    f_data[t_data.frame].append(copy.copy(t_data))
                except:
                    print(len(f_data), idx)
                    raise

                if t_data.track_id not in ids and t_data.obj_type != "dontcare":
                    ids.append(t_data.track_id)
                    n_trajectories += 1
                    n_in_seq += 1

            # only add existing frames
            n_trajectories_seq.append(n_in_seq)
            seq_data.append(f_data)
            f.close()

        if not loading_groundtruth:
            self.tracker = seq_data
            self.n_tr_trajectories = n_trajectories
            self.n_tr_seq = n_trajectories_seq
            if self.n_tr_trajectories == 0:
                return False
        else:
            # split ground truth and DontCare areas
            self.dcareas = []
            self.groundtruth = []
            # print(len(seq_data))
            for seq_idx in range(len(seq_data)):
                seq_gt = seq_data[seq_idx]
                s_g, s_dc = [], []
                for f in range(len(seq_gt)):
                    all_gt = seq_gt[f]
                    g, dc = [], []
                    for gg in all_gt:
                        if gg.obj_type == "dontcare":
                            dc.append(gg)
                        else:
                            g.append(gg)
                    s_g.append(g)
                    s_dc.append(dc)
                self.dcareas.append(s_dc)
                self.groundtruth.append(s_g)
            self.n_gt_seq = n_trajectories_seq
            self.n_gt_trajectories = n_trajectories

        return True

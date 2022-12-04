import cv2
import pandas as pd
import os
import numpy as np
from tqdm import tqdm


def non_max_suppression_post(outputs: np.ndarray, overlapThresh, counts=False):
    # if there are no boxes, return an empty list
    boxes = outputs.astype(float)
    # for out in outputs:
    #     x1, y1, x2, y2, = out
    #     fullOutputs.append([x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist(),
    #                         conf.tolist(), cls_conf.tolist()])
    # t = time.time()
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x2 = x1 + w
    y2 = y1 + h
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    cs = []
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

        # get the counts for each bounding box.
        cs.append(len(np.where(overlap > overlapThresh)[0]))

    if counts:
        return boxes[pick].astype(float), cs

    else:
        return boxes[pick].astype(float)


class CSV_Reader():
    frame_interval = 100   # Frame intervals to check for stagnant worms
    step = 1  # Frame step within each interval
    prevalence = 0.3  # Fraction of unchanged bbs needed to be considered "stagnant"
    padding = 100  # How many frames to pad by to be safe

    def __init__(self, csv, vid):
        """ Reads the csv and video and provides useful functions for determining
        time of death"""
        self.csv = csv
        self.vid = vid

        video = cv2.VideoCapture(vid)
        self.video = video
        self.frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        self.df = pd.read_csv(csv,
                              usecols=[0, 1, 2, 3, 4, 5],
                              names=["frame", "x", "y", "w", "h", "class"])

    def get_frame(self, frame_id):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, frame = self.video.read()
        return ret, frame

    def get_worms_from_frame(self, frame_id):
        """ Gets the frame image from a frame id,
        and then the bounding boxes associated with that image"""
        ret, frame = self.get_frame(frame_id)
        bbs = self.df[self.df["frame"] == frame_id]
        bbs = bbs.to_numpy()
        bbs = bbs[:, 1:5]
        return frame, bbs

    def get_worms_from_end(self, first: int, count: int, nms: float = 0.3):
        """Cycles through framse in forward from first to last, and fetches the
        bounding boxes.

        Args:
            first ([int]): [latest frame you go from]
            count ([int]): [how many frames forward to track]
            nms ([float]): [thresh of overlap for nms]

        Returns:
            list of tracked bounding boxes and all the bounding boxes
            in the selected frames.
        """
        last = first + count
        if (last > self.frame_count - 1):
            print("Please pick an earlier frame")
            last = self.frame_count - 1

        all_bbs = np.empty((0, 4), int)
        for i in range(first, last):
            _, bbs = self.get_worms_from_frame(i)
            all_bbs = np.append(all_bbs, bbs, axis=0)

        tracked = non_max_suppression_post(all_bbs, nms)
        self.tracked = tracked
        return tracked, all_bbs


if __name__ == "__main__":
    test_csv = "/mnt/sdb1/videos/resveratrol_data/csvs/1068.csv"
    test_vid = "/mnt/sdb1/videos/resveratrol_data/vids/1068.avi"

    def csv_reader_test():
        # Needs to validate outputs...
        test = CSV_Reader(test_csv, test_vid)
        a, b = test.get_frame(0)
        a, b = test.get_worms_from_frame(100)
        a, b = test.get_worms_from_end(1400, 10, 0.6)
        print("CSV_reader test passed!")

    csv_reader_test()

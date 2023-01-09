import cv2
import pandas as pd
import numpy as np


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


def draw_from_output(img, bbs, preds, col=(255, 255, 0), text=None):
    """ Img is cv2.imread(img) and bbs are (x1, y1, x2, y2, conf, cls_conf)
    Returns the image with all the boudning boxes drawn on the img """
    for output, pred in zip(bbs, preds):
        # output = [float(n) for n in output]
        if pred == 0:
            clss = "dead"
            col = (0, 0, 255)
        elif pred == 1:
            clss = "alive"
            col = (0, 255, 0)

        x1, y1, w, h = output
        x1, y1, w, h = int(x1), int(y1), int(w), int(h)
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), col, 2)

        if text is not None:
            cv2.putText(img, f"{clss}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)


class CSV_Reader():

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

    def get_worms_from_end(self, first: int, spread: int, nms: float = 0.3):
        """Cycles through framse in forward from first to last, and fetches the
        bounding boxes.

        Args:
            first ([int]): [latest frame you go from]
            spread ([int]): [how many frames forward to track]
            nms ([float]): [thresh of overlap for nms]

        Returns:
            list of tracked bounding boxes and all the bounding boxes
            in the selected frames.
        """
        last = first + spread
        if (last > self.frame_count - 1):
            print("Please pick an earlier frame")
            last = self.frame_count - 1

        all_bbs = np.empty((0, 4), int)
        for i in range(first, last):
            _, bbs = self.get_worms_from_frame(i)
            all_bbs = np.append(all_bbs, bbs, axis=0)

        tracked = non_max_suppression_post(all_bbs, nms)
        return tracked, all_bbs

    def fetch_worms(self, bbs: list, frame_id: int, pad=0, offset=(0, 0)):
        """Fetches worms in self.tracked by worm id on a given frame_id.
        Allows for padding and auto padding for worms that are skinny.
        Args:
            bbs (list): List of worm ids to be fetched.
            frame_id (int): Frame from which to fetch worms.
            pad (int, tuple): Padding in x and y direction. Tuple or Int.
            offset (tuple): Offset in x and y direction. Tuple.
            auto (tuple, optional): Auto pads for skinny worms.
        Returns:
            _type_: _description_
        """
        # Get pad ammount.
        if type(pad) == int:
            padX = pad
            padY = pad
        else:
            padX, padY = pad

        # Get the bbs for the frame.
        ret, frame = self.get_frame(frame_id)
        height, width = frame.shape[:2]

        if not ret:
            print(f"Frame {frame_id} not found.")
            pass
        # Get worm image for each frame
        worm_imgs = []
        for bb in bbs:
            x, y, w, h = bb.astype(int)
            x += offset[0]
            y += offset[1]
            x, y, w, h = x - padX, y - padY, w + 2*padX, h + 2*padY

            # Set x y lower and upper to account for padding / offset.
            y_l, y_u = max(0, y), min(height, y + h)
            x_l, x_u = max(0, x), min(width, x + w)

            worm_img = frame[y_l:y_u, x_l:x_u]
            worm_imgs.append(worm_img)
            # worm_img = frame[y:y+h, x:x+w]
            # worm_imgs.append(worm_img)

        return worm_imgs


class Series_Builder(CSV_Reader):
    """Takes the video and csv path and allows you to the frames for
    each serries within the video.=
    """
    def __init__(self, csv: str, vid: str, frequency=72, intervals=[32, 74], spread=5, nms=0.6, pad=5):
        """
        Args:
            csv (str): Path to the yolo csv file.
            vid (str): Path to the raw video file.
            frequency (int, optional): How often the video is sampled..
            interval (int, list[int]): (t-interval, t, t+interval).
            spread (int, optional): When grabbing multiple frames how many to grab.
            nms (float, optional): NMS for spread.
            pad (int): how much to pad the images by.
        """
        super().__init__(csv, vid)
        self.frequency = frequency
        # Set the option to have multiple intervals.
        if type(intervals) == int:
            self.intervals = [intervals]
        else:
            intervals.sort()
            self.intervals = intervals

        self.spread = spread
        self.nms = nms
        self.pad = pad

        # Builds the indecies.
        max_interval = max(self.intervals)
        self.start = 0 + max_interval  # First frame
        self.end = (self.frame_count - spread - max_interval) // max_interval * max_interval  # Last common multiple

        self.indecies = np.arange(self.start, self.end, self.frequency)

    def build_series(self, bbs, pre, cur, post):
        """Builds a series for t-1, t, t+1 for all the locations specified
        in bbs. Returns a array of [pre[], cur[], post[]].

        Args:
            bbs (_type_): _description_
            pre (_type_): _description_
            cur (_type_): _description_
            post (_type_): _description_
        """
        # Get pre current and post images for all bbs in bbs.
        PAD = self.pad
        pre_imgs = self.fetch_worms(bbs, pre, pad=PAD)
        cur_imgs = self.fetch_worms(bbs, cur, pad=PAD)
        post_imgs = self.fetch_worms(bbs, post, pad=PAD)
        assert len(pre_imgs) == len(cur_imgs) and len(cur_imgs) == len(post_imgs)

        # try:
        series = []
        for i in range(len(cur_imgs)):
            series.append([pre_imgs[i], cur_imgs[i], post_imgs[i]])

        return series

    def __len__(self):
        return len(self.indecies)

    def __getitem__(self, index: int):
        """Using the perams specified in init. Builds a bundle of image series
        that are used to determine the state of the worm. Returns series_list which
        contains a sublist of three image series and BBs which contains the
        "cur" bounding boxes specified.

        * series_list: is respective list of series. [index1, index2]
            index1: [[series1], [series2], ...]
            index2: [[series1], [series2], ...]

        Args:
            index (int): what "cur" frame to look at (index within indecies list).

        Returns:
            (list[list[imgs]], list[list[x, y, w, h]]):
        """
        cur = self.indecies[index]

        series_list = []
        for interval in self.intervals:
            pre, post = cur - interval, cur + interval
            # Gets the bounding boxes from the current frame.
            bbs, _ = self.get_worms_from_end(first=cur, spread=self.spread, nms=self.nms)

            series = self.build_series(bbs, pre, cur, post)
            series_list.append(series)

        return series_list, bbs

    def build_video(self, outputs: list, bbs: list, path: str):
        """Takes the predicted outputs, bounding boxes, and save path.
        Creates a video that lables the frames with the pred output and the bbs.

        Args:
            outputs (list): List of model outputs [frame1[w1, w2, w3], frame2[w1, w2, w3], ...]
            bbs (list): List of bounding boxes. First index is frame (like outputs).
            path (str): Save path for the video.
        """
        # TODO: Currently not using the input video shape in determining output shape.
        # w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # h = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fourcc, 3, (1920, 1080), True)

        # Check all the inputs match in length.
        assert len(bbs) == len(outputs) and len(bbs) == len(self.indecies)

        for i, idx in enumerate(self.indecies):
            _, frame = self.get_frame(idx)
            frame_preds = outputs[i]
            frame_bbs = bbs[i]

            draw_from_output(frame, bbs=frame_bbs, preds=frame_preds)
            writer.write(frame)

    @staticmethod
    def save_series(series: list, path: str):
        """Saves series of images to specified path.

        Args:
            series (list): list of 3 images.
            path (str): save path for the image.
        """
        try:
            stack = np.hstack(series)
            cv2.imwrite(path, stack)
            return True
        except Exception:
            print("Unable to create stack or save stack.")
            return False


if __name__ == "__main__":
    test_csv = "/mnt/sdb1/videos/resveratrol_data/csvs/1068.csv"
    test_vid = "/mnt/sdb1/videos/resveratrol_data/vids/1068.avi"

    def csv_reader_test():
        # Needs to validate outputs...
        test = CSV_Reader(test_csv, test_vid)
        a, b = test.get_frame(0)
        a, b = test.get_worms_from_frame(100)
        a, b = test.get_worms_from_end(1400, 10, 0.6)
        imgs = test.fetch_worms(a, 100, 2)
        assert len(imgs) == len(a)
        print("✅ CSV_reader test passed!")

    def build_series_test():
        test = Series_Builder(test_csv, test_vid, frequency=32, intervals=1)
        cur = test.indecies[0]
        pre, post = cur - test.intervals[0], cur + test.intervals[0]
        bbs, _ = test.get_worms_from_end(first=cur, spread=1, nms=0.98)
        all_series = test.build_series(bbs, pre, cur, post)
        test_save = test.save_series(all_series[2], "test.png")
        assert test_save is True
        print("✅ Build_series test passed")
        return all_series

    def get_index_test():
        test = Series_Builder(test_csv, test_vid, frequency=32, intervals=[10, 20])
        series_list, bbs = test[1]
        assert len(series_list) == 2
        print("✅ Get index test passed")

    def test_all():
        csv_reader_test()
        build_series_test()
        get_index_test()

        print("✅ All tests passed!")

    test_all()

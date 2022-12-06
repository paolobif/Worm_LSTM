import torch
import cv2
import numpy as np

from utils.yolo_utils import CSV_Reader
# from utils.training_utils import test_model
from utils.data_utils import LstmLoader
from model import ResnetLSTM

"""
Main file that runs through a a stack of videos and processes it using the
Yolo outputs in CSV format, and then makes time of death calls using
the LSTM.

From start to end of experiment.
    every 72 frames, run the pipeline.

Pipeline:
 - Get pre cur and post frames
 - Do NMS on N farmes to identify the regions of interest.
 - Check if worm has already been classifieid as dead.
 - Check if worm is next to any other worms in pre and post frame
 - Pass through lstm.

TODO:
 - When pulling fowrad and backward images pick 5 or so images and then look at
 the histograms or something to try and find the image that is closest to the mean.
"""


class series_builder(CSV_Reader):
    """Takes the video and csv path and allows you to the frames for
    each serries within the video.=
    """
    def __init__(self, csv, vid, interval=72, spread=5, nms=0.6):
        super().__init__(csv, vid)
        self.interval = interval
        self.spread = spread
        self.nms = nms

        # Builds the indecies.
        self.start = 0 + interval  # First frame
        self.end = (self.frame_count - spread) // interval * interval  # Last common multiple

        self.indecies = np.arange(self.start, self.end, self.interval)

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
        PAD = 5
        pre_imgs = self.fetch_worms(bbs, pre, pad=PAD)
        cur_imgs = self.fetch_worms(bbs, cur, pad=PAD)
        post_imgs = self.fetch_worms(bbs, post, pad=PAD)
        assert len(pre_imgs) == len(cur_imgs) and len(cur_imgs) == len(post_imgs)

        series = np.array([pre_imgs, cur_imgs, post_imgs], dtype=object)
        return series

    def __len__(self):
        return len(self.indecies)

    def __getitem__(self, index: int):
        cur = self.indecies[index]
        pre, post = cur - self.interval, cur + self.interval
        # Gets the bounding boxes from the current frame.
        bbs, _ = self.get_worms_from_end(first=cur, spread=self.spread, nms=self.nms)

        series = self.build_series(bbs, pre, cur, post)
        return series

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

    WEIGHTS = "weights/run0/weights12.pt"

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = ResnetLSTM().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    print("Succesfully laoded model to:", device)



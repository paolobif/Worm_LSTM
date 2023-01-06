import torch
import cv2
import numpy as np
import time

from tqdm import tqdm

from utils.utils import Series_Builder
from utils.data_utils import LstmLoader, transform
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

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def series_to_model_input(series: list):
    """Takes single series of 3 images as numpy array and converts
    it to a tensor stack for the model.

    Args:
        series (np.array): np array shape (3, x, y, 3)
    """
    assert len(series) == 3, "invalid series length"
    transformed = []
    for i in series:
        new = transform(i)
        new = new / 255
        transformed.append(new)

    stack = torch.stack(transformed)
    stack = stack.to(device=device)

    return stack


def multi_index_batching(series_list, bbs, model) -> list:
    """Takes a list of series built at different indecies ie. t+-30 and t+-10.
    Then from smallest interval to the highest. Anything classified as alive
    gets dropped and dead calls move to the next iterval.

    Args:
        series_list (list): List[t+-10[series, series], t+-20[series, series]]
        model (model): lstm model.

    Returns:
        list: classification for each worm.
    """
    pass


def process_bulk_series(generator, model) -> list:
    """Takes the series generator, and iterates through all of it passing
    each sub series through the LSTM. Returns alive and dead counts.

    - series is the list of stacks within a given frame interval.
    - within each series are the 3 image stacks for the interval.

    series --> [[[pre, cur, post], [pre, cur, post], ...], [...], [...]]
    """

    timeA = time.time()
    count = 0

    exp_preds = []
    exp_bbs = []
    for series_list, bbs in tqdm(generator):
        preds = []
        # series_count = len(all_series])
        # if len(all_series) == 0:
        #     continue

        for series in all_series:
            count += 1
            # print(series.shape)
            stack = series_to_model_input(series)
            output = model(stack)

            if output > 0.5:
                pred_class = 1
            elif output < 0.5:
                pred_class = 0

            preds.append(pred_class)

        exp_preds.append(preds)
        exp_bbs.append(bbs)

    timeB = time.time()

    print("Process took:", timeB - timeA)
    print(f"Over: {count} itterations.")
    return exp_preds, exp_bbs


if __name__ == "__main__":
    test_csv = "/mnt/sdb1/videos/resveratrol_data/csvs/1068.csv"
    test_vid = "/mnt/sdb1/videos/resveratrol_data/vids/1068.avi"

    WEIGHTS = "weights/run0/weights12.pt"

    model = ResnetLSTM().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    print("Succesfully laoded model to:", device)

    test = Series_Builder(test_csv, test_vid, intervals=[36], spread=1, nms=0.95)
    outputs = process_bulk_series(test, model)

from main import *

test_csv = "/mnt/sdb1/videos/resveratrol_data/csvs/1068.csv"
test_vid = "/mnt/sdb1/videos/resveratrol_data/vids/1068.avi"


def test_series_to_model_input():
    test = Series_Builder(test_csv, test_vid, interval=36, spread=3, nms=0.95)
    test_series, bbs = test[67]
    stack = series_to_model_input(test_series[0])
    assert len(stack) == 3
    assert float(torch.max(stack)) < 1


def test_all():
    test_series_to_model_input()


test_all()
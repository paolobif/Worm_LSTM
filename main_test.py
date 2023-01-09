from main import *

test_csv = "/mnt/sdb1/videos/resveratrol_data/csvs/1068.csv"
test_vid = "/mnt/sdb1/videos/resveratrol_data/vids/1068.avi"


def load_model():
    WEIGHTS = "weights/run0/weights12.pt"
    model = ResnetLSTM().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    return model


def test_series_to_model_input():
    intervals = [36, 72, 23]
    test = Series_Builder(test_csv, test_vid, intervals=intervals, spread=3, nms=0.95)
    test_series, bbs = test[10]
    assert len(test_series) == len(intervals)  # making sure intervals match.
    stack = series_to_model_input(test_series[0][0])
    assert len(stack) == 3
    assert float(torch.max(stack)) < 1
    print("✅  Series to Model Input Passed")


def test_multi_series_batching():
    model = load_model()
    test = Series_Builder(test_csv, test_vid, intervals=[30, 40, 50, 60], spread=3, nms=0.95)
    series_list, bbs = test[15]
    # Series list is two stacks of series at t+-36 and t+-72: [t1, t2].
    batch_results = multi_index_batching(series_list, model)
    assert len(batch_results) == len(series_list[0])
    print("✅  Multi Series Batching Passed")


def test_all():
    test_series_to_model_input()
    test_multi_series_batching()


test_all()
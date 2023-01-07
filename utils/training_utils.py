import torch
from sklearn.metrics import confusion_matrix

"""
Contains general tools needed for training and testing of the model.
"""


def test_model(model, data, device):
    """Passes already labled data through the model and returns
    and confusion matrix of: (tn, fp, fn, tp).

    1 --> alive
    0 --> dead

    Confusion Matrix
    tn: Dead correct
    fn: Dead incorrect
    tp: Alive correct
    fp: Alive incorrect

    Args:
        model (_type_): torch model.
        data (_type_): lstm datalaoder object.
        device (_type_): model device.

    Returns:
        _type_: _description_
    """
    with torch.no_grad():
        real = []
        pred = []
        for series, label in data:
            series = series.to(device)
            real_class = int(label)

            output = model(series)
            if output > 0.5:
                pred_class = 1
            elif output < 0.5:
                pred_class = 0

            real.append(real_class)
            pred.append(pred_class)

    # print(real, pred)
    matrix = confusion_matrix(real, pred).ravel()
    # Quick fix to account for when the test is perfect...
    if len(matrix) != 4:
        matrix = [1, 1, 1, 1]
    tn, fp, fn, tp = matrix
    return (tn, fp, fn, tp)


def convert_confusion(matrix):
    # Matrix is array of: [tn, fp, fn, tp]
    # Returns fractional accuracy for total, dead, and alive
    tn, fp, fn, tp = matrix
    print(matrix)
    accuracy = (tn + tp) / sum(matrix)
    dead_accuracy = tn / (tn + fn)
    alive_accuracy = tp / (fp + tp)
    return accuracy, dead_accuracy, alive_accuracy

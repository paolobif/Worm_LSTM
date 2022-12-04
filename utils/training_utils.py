import torch
from sklearn.metrics import confusion_matrix

"""
Contains general tools needed for training and testing of the model.
"""


def test_model(model, data, device):
    """Passes already labled data through the model and returns
    and confusion matrix of: (tn, fp, fn, tp).

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

    tn, fp, fn, tp = confusion_matrix(real, pred).ravel()
    return (tn, fp, fn, tp)

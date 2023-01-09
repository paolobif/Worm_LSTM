import torch

from utils.training_utils import test_model
from utils.data_utils import LstmLoader
from model import ResnetLSTM


def convert_confusion(matrix):
    # Matrix is array of: [tn, fp, fn, tp]
    # Returns fractional accuracy for total, dead, and alive
    tn, fp, fn, tp = matrix
    accuracy = (tn + tp) / sum(matrix)
    dead_accuracy = tn / (tn + fn)
    alive_accuracy = tp / (fp + tp)
    return accuracy, dead_accuracy, alive_accuracy


if __name__ == "__main__":
    WEIGHTS = "weights/training_f2/weights75.pt"
    DATA_PATH = "data/validation"
    # DATA_PATH = "data/validation"

    data = LstmLoader(DATA_PATH)
    print(f"{len(data)} - Samples Found.")

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = ResnetLSTM().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    print("Succesfully laoded model to:", device)

    test_accuracy = test_model(model=model, data=data, device=device)
    percentages = convert_confusion(test_accuracy)
    # print(f"""Overall Validation Accuracy: {test_accuracy[0]},
    # Alive Accuracy: {test_accuracy[1]},
    # wDead Acuracy: {test_accuracy[2]}""")
    print(percentages)

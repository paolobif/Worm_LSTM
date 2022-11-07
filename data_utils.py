import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import os
from skimage import io
import numpy as np


class LstmLoader(Dataset):
    def __init__(self, root_path, size=244, f_extract=None):
        """Loads the data for the the LSTM model that makes time of death calls.

        Args:
            root_path (str): Path to directory with Alive & Dead Folders
            size (int, optional): Input shape (size x size). Defaults to 244.
            f_extract (model): Pyotrch model that does the feature extraction step.
        """

        self.root_path = os.path.abspath(root_path)
        ALIVE = os.path.join(self.root_path, "Alive")
        DEAD = os.path.join(self.root_path, "Dead")

        # [[path_to_images, class], ..., ...]
        # Class:
        #  0: Dead
        #  1: Alive

        dead_seqs = [[os.path.join(DEAD, i), 0] for i in os.listdir(DEAD)]
        alive_seqs = [[os.path.join(ALIVE, i), 1] for i in os.listdir(ALIVE)]
        idxs = list(range(0, len(dead_seqs)))
        alive_seqs = [alive_seqs[i] for i in idxs]

        self.data = alive_seqs + dead_seqs
        self.alive = alive_seqs
        self.dead = dead_seqs

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5), (0.2))
        ])

        self.feature_extractor = f_extract

    def load_image(self, path, raw):
        image = io.imread(path)  # Load image

        if not raw:
            image = self.transform(image)  # Transorm image
            image = image / 255  # Scale values to 0 - 1

        if self.feature_extractor:
            image = self.feature_extractor(image)

        return image


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index, raw=False):
        # If raw == False:
        #   Returns the tensorfied and transformed image series
        # Else returns just the raw cv2 images.
        path, label = self.data[index]

        label = self.format_label(label)

        pre_img = self.load_image(os.path.join(path, "pre.jpg"), raw)
        cur_img = self.load_image(os.path.join(path, "cur.jpg"), raw)
        next_img = self.load_image(os.path.join(path, "next.jpg"), raw)

        if not raw:
            X = torch.stack([pre_img, cur_img, next_img])
        else:
            X = [pre_img, cur_img, next_img]

        return X, label

    @staticmethod
    def format_label(label):
        # Formats the label to one hot encode and
        # converts it to tensor.
#         if label == 0:
#             new_label = [1, 0]
#         elif label == 1:
#             new_label = [0, 1]

#         return torch.tensor(new_label)
        return torch.tensor(label)





if __name__ == "__main__":
    # labels = {"3223_day4_worm_16":1}
    # list_ids = ["3223_day4_worm_16"]
    f_path = "data"
    test = LstmLoader(f_path)
    print(test.__getitem__(5)[0].shape)

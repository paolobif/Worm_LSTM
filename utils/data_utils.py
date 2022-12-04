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
        Specific for training.

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
        # idxs = list(range(0, len(dead_seqs)))
        # alive_seqs = [alive_seqs[i] for i in idxs]

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


class LstmPostLoader(Dataset):
    def __init__(self, path:str, size=244) -> None:
        """Data laoder for data to be processed by the model. Takes a path
        to directory of sub directories with images [pre.jpg, cur.jpg, post.jpg].
        Returns appropriatly transformed images.

        Args:
            path (str): Path to directory with image stacks.
            size (int, optional): Size images are resized to. Defaults to 244.
        """

        self.path = path
        self.sub_dirs = self.check_files(path)  # List of sub directories.

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def load_image(self, path, raw):
        """Loads the image from the raw path."""
        image = io.imread(path)  # Load image

        if not raw:
            image = self.transform(image)  # Transorm image
            image = image / 255  # Scale values to 0 - 1

        return image

    def __len__(self):
        return len(self.sub_dirs)

    def __getitem__(self, index, raw=False):
        sub_dir = self.sub_dirs[index]
        path = os.path.join(self.path, sub_dir)

        pre_img = self.load_image(os.path.join(path, "pre.png"), raw)
        cur_img = self.load_image(os.path.join(path, "cur.png"), raw)
        next_img = self.load_image(os.path.join(path, "next.png"), raw)

        if not raw:
            stack = torch.stack([pre_img, cur_img, next_img])
        else:
            stack = [pre_img, cur_img, next_img]

        return stack

    @ staticmethod
    def check_files(path):
        """Checks to make sure all the subdir are valid
        And have "pre, cur, post" images.
        If not removes them from the list"""
        corrected_list = []
        sub_dirs = os.listdir(path)
        for sub_dir in sub_dirs:
            sub_path = os.path.join(path, sub_dir)
            files = os.listdir(sub_path)
            if "pre.png" not in files or "cur.png" not in files or "next.png" not in files:
                pass
            else:
                corrected_list.append(sub_dir)

        return corrected_list


if __name__ == "__main__":
    # labels = {"3223_day4_worm_16":1}
    # list_ids = ["3223_day4_worm_16"]
    def test_lsmt_loader():
        f_path = "data/training"
        test = LstmLoader(f_path)
        assert list(test[0][0].shape[2:4]) == [244, 244]
        print("LSTM Loader Test Passed!")

    # Test LstmPostLoader
    def test_lstm_post_loader():
        sample_path = "data/fake_samples"
        test2 = LstmPostLoader(sample_path)
        assert len(test2) == 3, "invalid file count. should be 3."
        assert list(test2[0].shape[2:4]) == [244, 244], "Invalid output shape."

        raw = test2.__getitem__(0, raw=True)
        assert len(raw) == 3, "unable to load raw images."
        print("LSTM Post Loader Test Passed!")

    test_lsmt_loader()
    test_lstm_post_loader()
import torch

from utils.training_utils import test_model
from utils.data_utils import LstmLoader
from model import ResnetLSTM

"""
Main file that runs through a a stack of videos and processes it using the
Yolo outputs in CSV format, and then makes time of death calls using
the LSTM.
"""


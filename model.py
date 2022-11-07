import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch


class Resnet18Pretrained(nn.Module):
    def __init__(self, **kwargs):
        super(Resnet18Pretrained, self).__init__()
        self.conv_model = models.resnet18(pretrained=True)

        for param in self.conv_model.parameters():
            param.requires_grad = True

        self.conv_model.fc = nn.Identity()

    def forward(self, x):
        # Output ([batch_size, 512])
        return self.conv_model(x)


class ResnetLSTM(nn.Module):
    def __init__(self, seq_len=3, **kwargs):
        super(ResnetLSTM, self).__init__()
        self.seq_len = 3
        self.hidden_size = 256
        self.cnn = Resnet18Pretrained()

        self.lstm = nn.LSTM(
            input_size=self.seq_len,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True)

        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, input):
        # batch_size, steps, H, W = input.shape
        # print(input.shape)
        x = self.cnn(input)
        x = x.view(1, 512, 3)
        x, (h_n, h_c) = self.lstm(x)
        x = self.linear(x[:, -1, :])
        x = torch.sigmoid(x)

        return x

# if __name__ = "__main__":

""" Module containing the Network definition used in the AIXI method.

Written by: Miquel Miró Nicolau (UIB), 2022
"""
import torch
from torch import nn


class Net(nn.Module):
    """Class that heritates from the pytorch module class and defines the Network used in the AIXI
    method.

    Args:
        num_channels (int): Number of channels of the input images.
        classes (int): Number of classes of the dataset.
    """

    def __init__(
        self, num_channels, classes, weights_path: str = None, do_sigmoid: bool = False
    ):
        # call the parent constructor
        super().__init__()

        self._do_simgoid = do_sigmoid

        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=25,
            kernel_size=(3, 3),
            padding="same",
        )
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(25)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(
            in_channels=25, out_channels=50, kernel_size=(3, 3), padding="same"
        )
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(50)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(in_features=32 * 32 * 50, out_features=25)  # 50
        self.dropout1 = nn.Dropout(p=0.5)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=25, out_features=15)
        self.dropout2 = nn.Dropout(p=0.5)
        self.relu5 = nn.ReLU()

        self.fc3 = nn.Linear(in_features=15, out_features=classes)

        if self._do_simgoid:
            self.sigmoid = nn.Sigmoid()

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu5(x)

        return self._end(x)

    def _end(self, x):
        x = self.fc3(x)

        if self._do_simgoid:
            x = self.sigmoid(x)

        return x


class OOD_Net(Net):
    def __init__(self, classes, *args, **kwargs):
        super().__init__(classes=classes, *args, **kwargs)

        self.h = nn.Linear(in_features=15, out_features=classes, bias=False)
        self.g_fc = nn.Linear(in_features=15, out_features=1)
        self.g_bn = nn.BatchNorm2d(1)
        self.g_sigmoid = nn.Sigmoid()

        self.fc3 = None

    def _end(self, x):
        h = self.h(x)

        g = self.g_fc(x)
        g = self.g_bn(g)
        g = torch.square(g)
        g = self.g_sigmoid(g)

        return super()._end(h / g)

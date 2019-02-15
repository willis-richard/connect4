from src.connect4.board import Board
from src.connect4.utils import Connect4Stats as info
from src.connect4.utils import NetworkStats as net_info

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Callable, Dict, Optional, Tuple

# import torch.nn.functional as F


# Input with N * channels * (6,7)
# Output with N * filters * (6,7)
convolutional_layer = \
    nn.Sequential(nn.Conv2d(in_channels=net_info.channels,
                            out_channels=net_info.filters,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            dilation=1,
                            groups=1,
                            bias=False),
                  nn.BatchNorm2d(net_info.filters),
                  nn.LeakyReLU())


# Input with N * filters * (6,7)
# Output with N * filters * (6,7)
class ResidualLayer(nn.Module):
    def __init__(self, filters=net_info.filters):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(filters)
        self.batch_norm2 = nn.BatchNorm2d(filters)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)

        out += residual
        out = self.relu(out)
        return out


# Input with N * filters * (6,7)
# Output with N * 1
class ValueHead(nn.Module):
    def __init__(self, filters=net_info.filters, fc_layers=net_info.n_fc_layers):
        super(ValueHead, self).__init__()
        self.conv1 = nn.Conv2d(filters, 1, 1)
        self.batch_norm = nn.BatchNorm2d(1)
        self.relu = nn.LeakyReLU()
        self.fcN = nn.Sequential(*[nn.Linear(net_info.area, net_info.area) for _ in range(fc_layers)])
        self.fc1 = nn.Linear(net_info.area, 1)
        self.tanh = torch.nn.Tanh()
        self.w1 = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.w2 = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = x.view(x.shape[0], 1, -1)
        x = self.fcN(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.tanh(x)
#         map from [-1, 1] to [0, 1]
        x = (x + self.w1) * self.w2
        x = x.view(-1, 1)
        return x


# Input with N * filters * (6,7)
# Output with N * 7
class PolicyHead(nn.Module):
    def __init__(self, filters=net_info.filters):
        super(PolicyHead, self).__init__()
        self.conv1 = nn.Conv2d(filters, 2, 1)
        self.batch_norm = nn.BatchNorm2d(2)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(2 * net_info.area, net_info.width)
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = x.view(x.shape[0], 1, -1)
        x = self.fc1(x)
        # x = self.softmax(x)
        # No idea why but if I had [[[ output then classifier bitched and wanted [[
        x = x.view(-1, net_info.width)
        return x


# Used in 8-ply testing
value_net= nn.Sequential(convolutional_layer,
                         nn.Sequential(*[ResidualLayer() for _ in range(net_info.n_residuals)]),
                         ValueHead())


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.body = nn.Conv2d(in_channels=net_info.channels,
        #                     out_channels=net_info.filters,
        #                     kernel_size=3,
        #                     stride=1,
        #                     padding=1,
        #                     dilation=1,
        #                     groups=1,
        #                     bias=False)
        self.body = nn.Sequential(convolutional_layer,
                                  nn.Sequential(*[ResidualLayer() for _ in range(net_info.n_residuals)]))
        self.value_head = ValueHead()
        # self.policy_head = PolicyHead()

    def forward(self, x):
        x = x.view(-1, net_info.channels, info.height, info.width)
        x = self.body(x)
        value = self.value_head(x)
        # policy = self.policy_head(x)
        return value, torch.ones(value.shape[0], net_info.width) #, policy


class Model():
    def __init__(self,
                 checkpoint: Optional[Dict] = None):
        self.net = Net()
        self.optimiser = optim.Adam(self.net.parameters())
        if checkpoint is not None:
            self.net.load_state_dict(checkpoint['net_state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device=torch.device("cpu")
        self.net.to(self.device)
        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.CrossEntropyLoss()

    def __call__(self, board: Board):
        board_tensor = board.to_tensor()
        board_tensor = board_tensor.to(self.device)
        return self.net(board_tensor)

    def criterion(self, x_value, x_policy, y_value, y_policy):
        value_loss = self.value_loss(x_value, y_value)
        # policy_loss = self.policy_loss(x_policy, y_policy)
        # L2 regularization loss is added via the optimiser (setting a weight_decay value)

        # return value_loss - policy_loss
        return value_loss
    # FIXME: How is the optimiser going to work?
    # https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
    # l2 loss https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/l2-regularization

    def train(self,
              data: DataLoader,
              n_epochs: int):
        for epoch in range(n_epochs):
            self.net.train()

            for board, y_value, y_policy in data:
                board = board.to(self.device)
                y_value = y_value.to(self.device)
                y_policy = y_policy.to(self.device)

                # zero the parameter gradients
                self.optimiser.zero_grad()

                # forward + backward + optimise
                x_value, x_policy = self.net(board)
                loss = self.criterion(x_value, x_policy, y_value, y_policy)
                loss.backward()
                self.optimiser.step()

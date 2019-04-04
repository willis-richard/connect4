from connect4.board import Board
from connect4.utils import Connect4Stats as info
from connect4.utils import NetworkStats as net_info

from connect4.neural.config import ModelConfig
from connect4.neural.stats import Stats

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR

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
        self.conv1 = nn.Conv2d(filters, 1, 1) # N*f*H*W -> N*1*H*W
        self.batch_norm = nn.BatchNorm2d(1)
        self.relu = nn.LeakyReLU()
        # in the linear we go from N*(H*W) -> N*(H*W)
        self.fcN = nn.Sequential(*[nn.Linear(net_info.area, net_info.area) for _ in range(fc_layers)])
        self.fc1 = nn.Linear(net_info.area, 1) # N*(H*W) -> N*(1)
        self.tanh = torch.nn.Tanh()
        self.w1 = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.w2 = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        # Flatten before linear layers
        x = x.view(x.shape[0], 1, -1)
        x = self.fcN(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.tanh(x)
#         map from [-1, 1] to [0, 1]
        x = (x + self.w1) * self.w2
        # FIXME: Is this needed?
        x = x.view(-1, 1)
        return x


# Input with N * filters * (6,7)
# Output with N * 7
class PolicyHead(nn.Module):
    def __init__(self, filters=net_info.filters):
        super(PolicyHead, self).__init__()
        self.conv1 = nn.Conv2d(filters, 2, 1) # N * f * H * W -> N * 2 * H * W
        self.batch_norm = nn.BatchNorm2d(2)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(2 * net_info.area, net_info.width) # N * f * (2 * H * W) -> N * f * W
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        # Must flatten before linear layer
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
        self.body = nn.Sequential(convolutional_layer,
                                  nn.Sequential(*[ResidualLayer() for _ in range(net_info.n_residuals)]))
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()

    def forward(self, x):
        # FIXME: Needed?
        x = x.view(-1, net_info.channels, info.height, info.width)
        x = self.body(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return value, policy


class ModelWrapper():
    def __init__(self,
                 config: ModelConfig,
                 file_name: Optional[str] = None):
        self.config = config
        self.net = Net()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.optimiser = optim.SGD(self.net.parameters(),
                                   lr=config.initial_lr,
                                   momentum=config.momentum,
                                   weight_decay=config.weight_decay)
        self.scheduler = MultiStepLR(self.optimiser,
                                     milestones=config.milestones,
                                     gamma=config.gamma)
        # self.optimiser = optim.Adam(self.net.parameters())
        if file_name is not None:
            checkpoint = torch.load(file_name)
            self.net.load_state_dict(checkpoint['net_state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            self.net.apply(weights_init)

        self.value_loss = nn.MSELoss()
        # FIXME: that this needs to be with logits, not just the class index
        # Google says: BCEWithLogitsLoss or MultiLabelSoftMarginLoss
        # self.policy_loss = nn.CrossEntropyLoss()
        self.policy_loss = nn.MultiLabelSoftMarginLoss()
        print("Constructed NN with {} parameters".format(sum(p.numel() for p in self.net.parameters() if p.requires_grad)))
        self.net.eval()
        # self.net.train(False)
        board_1 = Board()
        board_2 = Board()
        board_3 = Board()
        board_2.o_pieces=np.ones((info.height, info.width))
        board_3.x_pieces=np.ones((info.height, info.width))
        print("Test board output: empty board:  {}, full o board:  {}, full x board:  {}"
              .format(self.__call__(board_1), self.__call__(board_2), self.__call__(board_3)))

    def __call__(self, board: Board):
        board_tensor = torch.FloatTensor(board.to_array())
        board_tensor = board_tensor.to(self.device)
        value, prior = self.net(board_tensor)
        assert not torch.isnan(value).any()
        assert not torch.isnan(prior).any()
        value = value.cpu()
        value = value.view(-1)
        value = value.data.numpy()
        prior = prior.cpu()
        prior = prior.view(-1)
        prior = prior.data.numpy()
        return value, prior

    def save(self, file_name: str):
        torch.save(
            {
                'net_state_dict': self.net.state_dict(),
                'optimiser_state_dict': self.optimiser.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            },
            file_name)

    def criterion(self, x_value, x_policy, y_value, y_policy):
        value_loss = self.value_loss(x_value, y_value)
        policy_loss = self.policy_loss(x_policy, y_policy)
        # L2 regularization loss is added via the optimiser (setting a weight_decay value)

        return value_loss - policy_loss

    # FIXME: How is the optimiser going to work?
    # https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
    # l2 loss https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/l2-regularization

    def train(self,
              data,
              n_epochs: int):
        data = self.create_dataset(*data, batch_size=self.config.batch_size)
        self.net.train()
        for epoch in range(n_epochs):

            for board, y_value, y_policy in data:
                self.scheduler.step()
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
        # https://discuss.pytorch.org/t/output-always-the-same/5934/4
        # https://github.com/pytorch/pytorch/issues/5406
        # https://discuss.pytorch.org/t/model-eval-gives-incorrect-loss-for-model-with-batchnorm-layers/7561/13
        # Epic post in this one
        # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/33
        # self.net.train(False)
        self.net.eval()

    def evaluate_value_only(self, boards, values):
        # Note no policy here, 3rd arg unused
        data = self.create_dataset(boards, values, values, batch_size=4096)
        """Get an idea of how the initialisation is"""
        stats = Stats()

        with torch.set_grad_enabled(False):
            for board, value, _ in data:
                board, y_value = board.to(self.device), value.to(self.device)
                x_value, _ = self.net(board)
                loss = self.value_loss(x_value, y_value)
                # FIXME: flatten is the right way here yes?
                stats.update(x_value.numpy().flatten(), y_value.numpy().flatten(), loss.item())

        return stats

    def create_dataset(self, boards, values, policies, batch_size):
        data = Connect4Dataset(boards,
                               values,
                               policies)

        return DataLoader(data, batch_size=batch_size, shuffle=True)


def weights_init(m):
    return
    # classname = m.__class__.__name__
    # if classname.find('Conv2d') != -1:
    #     nn.init.constant_(m.weight, 1)
    # elif classname.find('BatchNorm2d') != -1:
    #     nn.init.constant_(m.weight, 1)
    #     nn.init.constant_(m.bias, 0)
    # elif classname.find('Linear') != -1:
    #     nn.init.constant_(m.weight, 1)
    #     nn.init.constant_(m.bias, 0)


class Connect4Dataset(Dataset):
    def __init__(self, boards, values, policies):
        assert len(boards) == len(values) == len(policies)
        self.boards = torch.FloatTensor(boards)
        self.values = torch.FloatTensor(values)
        self.policies = torch.FloatTensor(policies)

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx: int):
        return (self.boards[idx],
                self.values[idx],
                self.policies[idx])
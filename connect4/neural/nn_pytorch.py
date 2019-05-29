from connect4.board_c import Board
from connect4.utils import Connect4Stats as info
from connect4.utils import NetworkStats as net_info

from connect4.neural.config import ModelConfig
from connect4.neural.stats import CombinedStats, ValueStats
from connect4.neural.storage import GameStorage
from connect4.neural.training_game import GameData, TrainingData

import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR

from typing import (List,
                    Optional,
                    Sequence,
                    Union)


class Connect4Dataset(Dataset):
    def __init__(self,
                 boards: torch.FloatTensor,
                 values: torch.FloatTensor,
                 priors: torch.FloatTensor):
        self.boards = boards
        self.values = values
        self.priors = priors

    def save(self, filename):
        data = {}
        data['boards'] = self.boards
        data['values'] = self.values
        data['priors'] = self.priors

        torch.save(data, filename)

    @classmethod
    def load(cls, filename):
        data = torch.load(filename)
        return cls(data['boards'], data['values'], data['priors'])

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx: int):
        if self.priors is None:
            return (self.boards[idx],
                    self.values[idx])
        else:
            return (self.boards[idx],
                    self.values[idx],
                    self.priors[idx])


# Input with N * channels * (6,7)
# Output with N * filters * (6,7)
def create_convolutional_layer(in_channels=net_info.channels,
                               out_channels=net_info.filters):
    return nn.Sequential(nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   dilation=1,
                                   groups=1,
                                   bias=False),
                         nn.BatchNorm2d(out_channels),
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
    def __init__(self,
                 filters=net_info.filters,
                 fc_layers=net_info.n_fc_layers):
        super(ValueHead, self).__init__()
        self.conv1 = nn.Conv2d(filters, 1, 1)  # N*f*H*W -> N*1*H*W
        self.batch_norm = nn.BatchNorm2d(1)
        self.relu = nn.LeakyReLU()
        # in the linear we go from N*(H*W) -> N*(H*W)
        self.fcN = nn.Sequential(*[nn.Linear(net_info.area, net_info.area) for _ in range(fc_layers)])
        self.fc1 = nn.Linear(net_info.area, 1)  # N*(H*W) -> N*(1)
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
        x = x.view(-1)
        return x


# Input with N * filters * (6,7)
# Output with N * 7
class PolicyHead(nn.Module):
    def __init__(self, filters=net_info.filters):
        super(PolicyHead, self).__init__()
        self.conv1 = nn.Conv2d(filters, 2, 1)  # N * f * H * W -> N * 2 * H * W
        self.batch_norm = nn.BatchNorm2d(2)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(2 * net_info.area, net_info.width)  # N * f * (2 * H * W) -> N * f * W
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        # Must flatten before linear layer
        x = x.view(x.shape[0], 1, -1)
        x = self.fc1(x)
        # No idea why but if I had [[[ output then classifier bitched and wanted [[
        x = x.view(-1, net_info.width)
        x = self.softmax(x)
        return x


# Used in 8-ply testing
def build_value_net(filters=net_info.filters,
                    n_residual_layers=net_info.n_residuals,
                    value_head_fc_layers=net_info.n_fc_layers):
    value_net = nn.Sequential(create_convolutional_layer(2, filters),
                              nn.Sequential(*[ResidualLayer(filters) for _ in range(n_residual_layers)]),
                              ValueHead(filters, value_head_fc_layers))
    return value_net


def build_policy_net(filters=net_info.filters,
                     n_residual_layers=net_info.n_residuals):
    policy_net = nn.Sequential(create_convolutional_layer(2, filters),
                               nn.Sequential(*[ResidualLayer(filters) for _ in range(n_residual_layers)]),
                               PolicyHead(filters))
    return policy_net


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.body = nn.Sequential(create_convolutional_layer(),
                                  nn.Sequential(*[ResidualLayer() for _ in range(net_info.n_residuals)]))
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()

    def forward(self, x):
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
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.net.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.optimiser = optim.SGD(self.net.parameters(),
                                   lr=config.initial_lr,
                                   momentum=config.momentum,
                                   weight_decay=config.weight_decay)
        # self.optimiser = optim.Adam(self.net.parameters())
        self.scheduler = MultiStepLR(self.optimiser,
                                     milestones=config.milestones,
                                     gamma=config.gamma)
        if file_name is not None:
            checkpoint = torch.load(file_name)
            self.net.load_state_dict(checkpoint['net_state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # else:
        #     self.net.apply(weights_init)

        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.BCELoss()
        # FIXME: that this needs to be with logits, not just the class index
        # Google says: BCEWithLogitsLoss or MultiLabelSoftMarginLoss
        # self.policy_loss = nn.CrossEntropyLoss()
        # self.policy_loss = nn.MultiLabelSoftMarginLoss()
        print("Constructed NN with {} parameters".format(sum(p.numel() for p in self.net.parameters() if p.requires_grad)))
        self.net.eval()

    def __call__(self, input_: Union[Board, List[Board]]):
        if isinstance(input_, Board):
            return self.call_board(input_)
        elif isinstance(input_, list):
            return self.call_list(input_)
        else:
            raise TypeError('ModelWrapper called with {}. It accepts either a Board nor a list(Board)'.format(type(input_)))

    def call_board(self, board: Board):
        board_tensor = torch.FloatTensor(board.to_array())
        board_tensor = board_tensor.view(1, *board_tensor.size())
        board_tensor = board_tensor.to(self.device)
        value, prior = self.net(board_tensor)
        try:
            assert not torch.isnan(value).any()
            assert not torch.isnan(prior).any()
        except AssertionError:
            print(board, value, prior)
            assert False
        value = value.cpu().view(-1).data.numpy()
        prior = prior.cpu().view(-1).data.numpy()
        return value, prior

    def call_list(self, board_list: List[Board]):
        board_tensor = torch.FloatTensor(list(map(lambda x: x.to_array(),
                                                  board_list)))
        board_tensor = board_tensor.to(self.device)
        values, priors = self.net(board_tensor)
        try:
            assert not torch.isnan(values).any()
            assert not torch.isnan(priors).any()
        except AssertionError:
            print(board_tensor, values, priors)
            assert False
        values = values.cpu().data.numpy()
        priors = priors.cpu().data.numpy()
        return values, priors

    def save(self, file_name: str):
        torch.save(
            {
                'net_state_dict': self.net.state_dict(),
                'optimiser_state_dict': self.optimiser.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            },
            file_name)

    def criterion(self, x_value, x_policy, y_value, y_policy):
        assert x_value.shape == y_value.shape
        assert x_policy.shape == y_policy.shape

        value_loss = self.value_loss(x_value, y_value)
        policy_loss = self.policy_loss(x_policy, y_policy)
        # L2 regularization loss is added via the optimiser (setting a weight_decay value)

        return value_loss + policy_loss

    # FIXME: How is the optimiser going to work?
    # https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
    # l2 loss https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/l2-regularization

    def train(self, data: Connect4Dataset):
        data = DataLoader(data,
                          batch_size=self.config.batch_size,
                          shuffle=True)
        self.net.train()
        for epoch in range(self.config.n_training_epochs):
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
        # Perhaps it is to do with the batchnorm: https://arxiv.org/abs/1702.03275
        self.net.eval()

    def evaluate_value_only(self, data: Connect4Dataset):
        # Note no policy here, 3rd arg unused
        data = DataLoader(data, batch_size=4096, shuffle=True)
        """Get an idea of how the initialisation is"""
        stats = ValueStats()

        with torch.set_grad_enabled(False):
            for board, value in data:
                board, y_value = board.to(self.device), value.to(self.device)
                x_value, _ = self.net(board)
                loss = self.value_loss(x_value, y_value)
                assert x_value.shape == y_value.shape
                # FIXME: flatten is the right way here yes?
                stats.update(x_value.cpu().numpy().flatten(),
                             y_value.cpu().numpy().flatten(),
                             loss.item())
        return stats

    def evaluate(self,
                 data: Connect4Dataset,
                 batch_size: int = 4096,
                 shuffle: bool = True):
        test_gen = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        return evaluate(test_gen,
                        self.net,
                        self.device,
                        self.value_loss,
                        self.policy_loss)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.constant_(m.weight, 0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight, 0)
        nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        nn.init.constant_(m.weight, 0)
        nn.init.constant_(m.bias, 0)


def evaluate(train_gen, net, device, value_criterion, prior_criterion):
    stats = CombinedStats()

    for board, value, prior in train_gen:
        board, value, prior = board.to(device), value.to(device), prior.to(device)

    value_output, prior_output = net(board)
    assert value_output.shape == value.shape
    assert prior_output.shape == prior.shape

    value_loss = value_criterion(value_output, value)
    prior_loss = prior_criterion(prior_output, prior)

    stats.update(value_output.cpu().detach().numpy(),
                 value.cpu().numpy(),
                 value_loss,
                 prior_output.cpu().detach().numpy(),
                 prior.cpu().numpy(),
                 prior_loss)

    return stats


class TrainingDataStorage(GameStorage):
    def __init__(self, folder_path: str):
        super().__init__(folder_path)

    def td_file_name(self, iteration):
        return "{}/data/{}.pth".format(self.folder_path, iteration)

    def save(self, games: List[GameData]):
        super().save(games)

        data = np.sum(list(map(lambda x: x.data, games)))
        board_t, value_t, prior_t = native_to_pytorch(data.boards,
                                                      data.values,
                                                      data.priors,
                                                      add_fliplr=True)

        dataset = Connect4Dataset(board_t, value_t, prior_t)
        dataset.save(self.td_file_name(self.iteration))

    def get_dataset(self):
        n = min(20, int((self.iteration + 1) / 2))
        file_names = [self.td_file_name(i)
                      for i in range(self.iteration,
                                     self.iteration - n,
                                     -1)]
        return ConcatDataset([Connect4Dataset.load(f)
                              for f in file_names])


def native_to_pytorch(boards: List[Board],
                      values: Sequence[float],
                      priors: List[Sequence[float]] = None,
                      to_move_channel: bool = True,
                      add_fliplr: bool = False):
    assert len(boards) == len(values)
    if add_fliplr:
        flip_boards = list(map(lambda x: x.create_fliplr(), boards))
        boards.extend(flip_boards)
        values = np.concatenate((values, values), axis=None)
    boards = torch.FloatTensor(list(map(
        lambda x: x.to_array(), boards)))
    if not to_move_channel:
        boards = boards[:, 1:]
    values = torch.FloatTensor(values)
    if priors is None:
        priors = None
    else:
        if add_fliplr:
            flip_priors = list(map(lambda x: np.flip(x), priors))
            priors.extend(flip_priors)
        assert len(boards) == len(priors)
        priors = torch.FloatTensor(priors)

    return boards, values, priors

from oinkoink.board import Board
from oinkoink.utils import Connect4Stats as info

from oinkoink.neural.config import ModelConfig, NetConfig
from oinkoink.neural.stats import CombinedStats, ValueStats

from oinkoink.neural.pytorch.data import Connect4Dataset

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from typing import List, Optional, Union


# Input with N * channels * (6,7)
# Output with N * filters * (6,7)
def create_convolutional_layer(in_channels,
                               out_channels):
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
    def __init__(self, filters):
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
                 filters: int,
                 fc_layers: int):
        super(ValueHead, self).__init__()
        self.conv1 = nn.Conv2d(filters, 1, 1)  # N*f*H*W -> N*1*H*W
        self.batch_norm = nn.BatchNorm2d(1)
        self.relu = nn.LeakyReLU()
        # in the linear we go from N*(H*W) -> N*(H*W)
        self.fcN = nn.Sequential(*[nn.Linear(info.area, info.area)
                                   for _ in range(fc_layers)])
        # N*(H*W) -> N*(1)
        self.fc1 = nn.Linear(info.area, 1)
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
    def __init__(self, filters: int):
        super(PolicyHead, self).__init__()
        # N * f * H * W -> N * 2 * H * W
        self.conv1 = nn.Conv2d(filters, 2, 1)
        self.batch_norm = nn.BatchNorm2d(2)
        self.relu = nn.LeakyReLU()
        # N * f * (2 * H * W) -> N * f * W
        self.fc1 = nn.Linear(2 * info.area, info.width)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        # Must flatten before linear layer
        x = x.view(x.shape[0], 1, -1)
        x = self.fc1(x)
        # No idea why but if I had [[[ output then classifier bitched and wanted [[
        x = x.view(-1, info.width)
        x = self.softmax(x)
        return x


class Net(nn.Module):
    def __init__(self, config: NetConfig = NetConfig()):
        super(Net, self).__init__()
        self.body = nn.Sequential(
            create_convolutional_layer(config.channels, config.filters),
            nn.Sequential(*[ResidualLayer(config.filters)
                            for _ in range(config.n_residuals)]))
        self.value_head = ValueHead(config.filters, config.n_fc_layers)
        self.policy_head = PolicyHead(config.filters)

    def forward(self, x):
        x = self.body(x)
        value = self.value_head(x)
        prior = self.policy_head(x)
        return value, prior


class ModelWrapper():
    def __init__(self,
                 config: ModelConfig,
                 file_name: Optional[str] = None):
        self.config = config
        self.net = Net(config.net_config)
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
            checkpoint = torch.load(file_name, self.device)
            self.net.load_state_dict(checkpoint['net_state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # else:
        #     self.net.apply(weights_init)

        self.value_loss = nn.MSELoss()
        self.prior_loss = nn.BCELoss()
        print("Constructed NN with {} parameters".format(
            sum(p.numel() for p in self.net.parameters() if p.requires_grad)))
        self.net.eval()

    def __call__(self, input_: Union[Board, List[Board]]):
        if isinstance(input_, Board):
            return self._call_board(input_)
        elif isinstance(input_, list):
            return self._call_list(input_)
        else:
            raise TypeError('ModelWrapper called with {}. It accepts either a '
                            'Board nor a list(Board)'.format(type(input_)))

    def evaluate(self,
                 data: Connect4Dataset,
                 batch_size: int = 4096,
                 shuffle: bool = True):
        test_gen = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

        return evaluate(test_gen,
                        self.net,
                        self.device,
                        self.value_loss,
                        self.prior_loss)

    def evaluate_value_only(self, data: Connect4Dataset):
        test_gen = DataLoader(data, batch_size=4096, shuffle=True)

        return evaluate_value_only(test_gen,
                                   self.net,
                                   self.device,
                                   self.value_loss)

    def train(self,
              data: Connect4Dataset,
              print_stats: bool = False):
        data = DataLoader(data,
                          batch_size=self.config.batch_size,
                          shuffle=True)
        self.net.train()
        for epoch in range(self.config.n_training_epochs):
            if print_stats:
                stats = CombinedStats()
            for board, y_value, y_prior in data:
                board = board.to(self.device)
                y_value = y_value.to(self.device)
                y_prior = y_prior.to(self.device)

                # zero the parameter gradients
                self.optimiser.zero_grad()

                # forward + backward + optimise
                x_value, x_prior = self.net(board)

                value_loss, prior_loss = self._criterion(x_value,
                                                         x_prior,
                                                         y_value,
                                                         y_prior)
                loss = value_loss + prior_loss
                loss.backward()
                self.optimiser.step()
                if print_stats:
                    stats.update(x_value.cpu().detach().numpy(),
                                 y_value.cpu().numpy(),
                                 value_loss,
                                 x_prior.cpu().detach().numpy(),
                                 y_prior.cpu().numpy(),
                                 prior_loss)

            if print_stats:
                print("epoch: {}\n{}".format(epoch, stats))

        self.scheduler.step()
        self.net.eval()

    def save(self, folder_path: str):
        torch.save(
            {
                'net_state_dict': self.net.state_dict(),
                # 'optimiser_state_dict': self.optimiser.state_dict()
                'optimiser_state_dict': self.optimiser.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            },
            folder_path + '/net.pth')

    def _call_board(self, board: Board):
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

    def _call_list(self, board_list: List[Board]):
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

        return values.cpu().data.numpy(), priors.cpu().data.numpy()

    def _criterion(self, x_value, x_prior, y_value, y_prior):
        assert x_value.shape == y_value.shape
        assert x_prior.shape == y_prior.shape

        value_loss = self.value_loss(x_value, y_value)
        prior_loss = self.prior_loss(x_prior, y_prior)
        # L2 regularization loss is added via the optimiser (setting a weight_decay value)

        return value_loss, prior_loss


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


def evaluate(test_gen, net, device, value_criterion, prior_criterion):
    stats = CombinedStats()

    with torch.set_grad_enabled(False):
        for board, value, prior in test_gen:
            board, value, prior = board.to(device), value.to(device), prior.to(device)

            value_output, prior_output = net(board)
            assert value_output.shape == value.shape
            assert prior_output.shape == prior.shape

            value_loss = value_criterion(value_output, value)
            prior_loss = prior_criterion(prior_output, prior)

            stats.update(value_output.cpu().numpy(),
                         value.cpu().numpy(),
                         value_loss,
                         prior_output.cpu().numpy(),
                         prior.cpu().numpy(),
                         prior_loss)

    return stats

def evaluate_value_only(test_gen, net, device, value_criterion):
    stats = ValueStats()

    with torch.set_grad_enabled(False):
        for board, value in test_gen:
            board, y_value = board.to(device), value.to(device)
            x_value, _ = net(board)
            loss = value_criterion(x_value, y_value)
            assert x_value.shape == y_value.shape
            stats.update(x_value.cpu().numpy(),
                         y_value.cpu().numpy(),
                         loss.item())
    return stats

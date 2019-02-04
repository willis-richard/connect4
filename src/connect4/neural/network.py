from src.connect4.utils import NetworkStats as net

import torch
import torch.nn as nn

# import torch.nn.functional as F


# Input with N * channels * (6,7)
# Output with N * filters * (6,7)
convolutional_layer = \
    nn.Sequential(nn.Conv2d(in_channels=net.channels,
                            out_channels=net.filters,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            dilation=1,
                            groups=1,
                            bias=False),
                  nn.BatchNorm2d(net.filters),
                  nn.LeakyReLU())


# Input with N * filters * (6,7)
# Output with N * filters * (6,7)
class ResidualLayer(nn.Module):
    def __init__(self, filters=net.filters):
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
# Output with N * area
class ValueHead(nn.Module):
    def __init__(self, filters=net.filters, fc_layers=net.n_fc_layers):
        super(ValueHead, self).__init__()
        self.conv1 = nn.Conv2d(filters, 1, 1)
        self.batch_norm = nn.BatchNorm2d(1)
        self.relu = nn.LeakyReLU()
        self.fcN = nn.Sequential(*[nn.Linear(net.area, net.area) for _ in range(n_fc_layers)])

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = x.view(x.shape[0], 1, -1)
        x = self.fcN(x)
        x = self.relu(x)
        return x


# Input with N * area
# Output with N * 1
class ValueTip(nn.Module):
    def __init__(self):
        super(ValueTip, self).__init__()
        self.fc1 = nn.Linear(net.area, 1)
        self.tanh = torch.nn.Tanh()
        self.w1 = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.w2 = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
#         map from [-1, 1] to [0, 1]
        x = (x + self.w1) * self.w2
        x = x.view(-1, 1)
        return x


# Input with N * area
# Output with N * 3 (ready for NLLLoss)
class ClassifierTip(nn.Module):
    def __init__(self):
        super(ClassifierTip, self).__init__()
        self.fc1 = nn.Linear(net.area, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 3)
        # x = nn.Softmax(x)
        x = nn.LogSoftmax(x)
        return x


missing_tip = nn.Sequential(convolutional_layer,
                            nn.Sequential(*[ResidualLayer() for _ in range(net.n_residuals)]),
                            ValueHead())

value_net = nn.Sequential(missing_tip, ValueTip())
classifier_net = nn.Sequential(missing_tip, ClassifierTip())

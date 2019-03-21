from connect4.board import Board
from connect4.utils import Connect4Stats as info
from connect4.utils import NetworkStats as net_info

from connect4.neural.config import ModelConfig

from tensorflow.keras.layers import (Activation,
                          add,
                          BatchNormalization,
                          Conv2D,
                          Dense,
                          Input)
from tensorflow.keras.models import Model, Sequential

from typing import Callable, Dict, Optional, Tuple


# N = batch size
# Input with N * channels * (6,7)
# Output with N * filters * (6,7)
convolutional_layer = \
    model = Sequential([Conv2D(filters=net_info.filters,
                        kernel_size=3,
                        strides=1,
                        padding='valid',
                        data_format='channels_first', # to be consistent with pytorch
                        activation=None,
                        use_bias=False),
                  BatchNormalization(axis=1), # due to channels_first
                  Activation('relu')]) # what about alpha=0.01 for LeakyRelu consistency?


# Input with N * filters * (6,7)
# Output with N * filters * (6,7)
class ResidualLayer(Model):
    def __init__(self, filters=net_info.filters):
        super(ResidualLayer, self).__init__()
        self.conv1 = Conv2D(filters=filters, kernel_size=3, use_bias=False)
        self.conv2 = Conv2D(filters=filters, kernel_size=3, use_bias=False)
        self.batch_norm1 = BatchNormalization(axis=1)
        self.batch_norm2 = BatchNormalization(axis=1)
        self.relu = Activation('relu')

    def call(self, x):
        residual = x
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)

        out = add([residual, y])
        out = self.relu(out)
        return out


# Input with N * filters * (6,7)
# Output with N * 1
class ValueHead(Model):
    def __init__(self, filters=net_info.filters, fc_layers=net_info.n_fc_layers):
        super(ValueHead, self).__init__()
        self.conv1 = Conv2D(filters=1, kernel_size=1)
        self.batch_norm = BatchNormalization(axis=1)
        self.relu = Activation('relu')
        self.fcN = Sequential([Dense(net_info.area, input_shape=(net_info.area,)) for _ in range(fc_layers)])
        self.fc1 = Dense(1, input_shape=(net_info.area,))
        self.tanh = Activation('tanh')
        # self.w1 = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        # self.w2 = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def call(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        # x = x.view(x.shape[0], 1, -1)
        x = self.fcN(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.tanh(x)
#         map from [-1, 1] to [0, 1]
        x = (x + 1.0) * 0.5
        # x = x.view(-1, 1)
        return x


# Input with N * filters * (6,7)
# Output with N * 7
class PolicyHead(Model):
    def __init__(self):
        super(PolicyHead, self).__init__()
        self.conv1 = Conv2D(filters=2, kernel_size=1)
        self.batch_norm = BatchNormalization(axis=1)
        self.relu = Activation('relu')
        self.fc1 = Dense(net_info.width, input_shape=(2 * net_info.area))

    def call(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        # x = x.view(x.shape[0], 1, -1)
        x = self.fc1(x)
        # x = x.view(-1, net_info.width)
        return x


# Used in 8-ply testing
value_net = Sequential([convolutional_layer,
                       Sequential([ResidualLayer() for _ in range(net_info.n_residuals)]),
                       ValueHead()])


class Net(Model):
    def __init__(self):
        super(Net, self).__init__()
        self.body = Sequential([convolutional_layer,
                                Sequential([ResidualLayer() for _ in range(net_info.n_residuals)])])
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()

    def call(self, x):
        x = x.view(-1, net_info.channels, info.height, info.width)
        x = self.body(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return [value, policy]


class Model():
    def __init__(self,
                 config: ModelConfig,
                 checkpoint: Optional[Dict] = None):
        self.config = config

        # build body, build two tips?
        # https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
        inputs = Inputs((info.width, info.height, 3))

        self.net = Net()

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimiser = optim.SGD(self.net.parameters(),
                                   lr=config.initial_lr,
                                   momentum=config.momentum,
                                   weight_decay=config.weight_decay)
        self.scheduler = MultiStepLR(self.optimiser,
                                     milestones=config.milestones,
                                     gamma=config.gamma)
        # self.optimiser = optim.Adam(self.net.parameters())
        if checkpoint is not None:
            self.net.load_state_dict(checkpoint['net_state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            # Initialise weights to zero (output should be 0.5)
            self.net.apply(weights_init)
        self.value_loss = nn.MSELoss()
        # FIXME: that this needs to be with logits, not just the class index
        # Google says: BCEWithLogitsLoss or MultiLabelSoftMarginLoss
        # self.policy_loss = nn.CrossEntropyLoss()
        self.policy_loss = nn.MultiLabelSoftMarginLoss()
        print("Constructed NN with {} parameters".format(sum(p.numel() for p in self.net.parameters() if p.requires_grad)))
        self.net.eval()
        # self.net.train(False)

    def __call__(self, board: Board):
        board_tensor = board.to_tensor()
        board_tensor = board_tensor.to(self.device)
        return self.net(board_tensor)

    def criterion(self, x_value, x_policy, y_value, y_policy):
        value_loss = self.value_loss(x_value, y_value)
        policy_loss = self.policy_loss(x_policy, y_policy)
        # L2 regularization loss is added via the optimiser (setting a weight_decay value)

        return value_loss - policy_loss

    # FIXME: How is the optimiser going to work?
    # https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
    # l2 loss https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/l2-regularization

    def train(self,
              data: DataLoader,
              n_epochs: int):
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

from connect4.board import Board
from connect4.utils import Connect4Stats as info
from connect4.utils import NetworkStats as net_info

from connect4.neural.config import ModelConfig
from connect4.neural.stats import Stats

import numpy as np
from tensorflow.keras.initializers import Constant, Ones
from tensorflow.keras.layers import (Activation,
                          add,
                          BatchNormalization,
                          Conv2D,
                          Dense,
                          Input,
                          Layer,
                          Reshape)
import tensorflow.keras.losses
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence
from typing import Callable, Dict, Optional, Tuple

# FIXME: CPU tensorflow cannot do NCHW right now. Added move axis_hack and set
# steps to undo:
# uncomment data_format lines in conv2D
# change axis=3 to axis=1 in batch norm layers
# delete np.moveaxis lines
# revert Inputs to NCHW

# N = batch size
# Input with N * channels * (6,7)
# Output with N * filters * (6,7)
class ConvolutionalLayer():
    def __init__(self, weight_decay: float, filters: int=net_info.filters):
        self.conv = Conv2D(filters=filters,
                           kernel_size=3,
                           strides=(1, 1),
                           padding='same',
                           # data_format='channels_first', # to be consistent with pytorch
                           activation=None,
                           use_bias=False,
                           kernel_regularizer=l2(weight_decay))
        self.batch_norm = BatchNormalization(axis=3) # due to channels_first
        self.relu = Activation('relu') # what about alpha=0.01 for LeakyRelu consistency?

    def build(self, input_: Input):
        x = self.conv(input_)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


# Input with N * filters * (6,7)
# Output with N * filters * (6,7)
class ResidualLayer():
    def __init__(self, weight_decay: float, filters: int=net_info.filters):
        self.conv1 = Conv2D(filters=filters,
                            kernel_size=3,
                            padding='same',
                            use_bias=False,
                            # data_format='channels_first',
                            kernel_regularizer=l2(weight_decay))
        self.conv2 = Conv2D(filters=filters,
                            kernel_size=3,
                            padding='same',
                            use_bias=False,
                            # data_format='channels_first',
                            kernel_regularizer=l2(weight_decay))
        self.batch_norm1 = BatchNormalization(axis=3)
        self.batch_norm2 = BatchNormalization(axis=3)
        self.relu = Activation('relu')

    def build(self, input_: Input):
        residual = input_
        out = self.conv1(input_)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)

        out = add([residual, out])
        out = self.relu(out)
        return out


# Actually these only need to be functions?
# https://towardsdatascience.com/understanding-residual-networks-9add4b664b03
class Body():
    def __init__(self, weight_decay: float, n_residuals: int=net_info.n_residuals):
        self.conv = ConvolutionalLayer(weight_decay)
        self.residuals = [ResidualLayer(weight_decay) for _ in range(n_residuals)]

    def build(self, input_: Input):
        x = self.conv.build(input_)
        for r in self .residuals:
            x = r.build(x)
        return x


# Input with N * filters * (6,7)
# Output with N * 1
class ValueHead():
    # class AddOneDivideByTwo(Layer):
    #     def __init__(self, **kwargs):
    #        super(AddOneDivideByTwo, self).__init__(**kwargs)

    #     def build(self, input_shape):
    #         # Create a trainable weight variable for this layer.
    #         # self.kernel = self.add_weight(name='kernel',
    #         #                               shape=(input_shape[1], 1),
    #         #                               initializer=keras.initializers.Constant(value=0.5),
    #         #                               trainable=False)
    #         super(AddOneDivideByTwo, self).build(input_shape)  # Be sure to call this at the end

    #     def call(self, x):
    #         x = x + 1
    #         x = x * 0.5
    #         # return K.dot(x, self.kernel)
    #         return x

    #     def compute_output_shape(self, input_shape):
    #         # assert input_shape is batchsize by 1
    #         assert len(input_shape) == 2
    #         assert input_shape[1] == 1
    #         return input_shape[0]

    def __init__(self,
                 weight_decay: float,
                 filters: int=net_info.filters,
                 fc_layers: int=net_info.n_fc_layers):
        self.conv1 = Conv2D(filters=1,
                            kernel_size=1,
                            # data_format='channels_first',
                            kernel_regularizer=l2(weight_decay))
        self.batch_norm = BatchNormalization(axis=3)
        self.relu = Activation('relu')
        # Keeps the 'area' of the input, but... only 8 params in the model definition...
        self.fcN = Sequential([Dense(net_info.area, kernel_regularizer=l2(weight_decay)) for _ in range(fc_layers)])
        self.fc1 = Dense(1, kernel_regularizer=l2(weight_decay))
        self.tanh = Activation('tanh')
        # N.B could have used two untrainable Dense layers, one that adds bias of 1, then one that multiplies by 0.5, no bias
        # https://keras.io/layers/core/
        # self.final_layer = AddOneDivideByTwo(trainable=False)
        self.add_one = Dense(1, trainable=False, kernel_initializer='zeros', use_bias=True, bias_initializer=Ones())
        self.divide_by_two = Dense(1, trainable=False, kernel_initializer=Constant(value=0.5), use_bias=False)

    def build(self, input_: Input):
        x = self.conv1(input_)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = Reshape((net_info.area,))(x)
        x = self.fcN(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.tanh(x)
#         map from [-1, 1] to [0, 1]
        # x = (x + 1.0) * 0.5
        # x = self.final_layer(x)
        x = self.add_one(x)
        x = self.divide_by_two(x)
        # x = x.view(-1, 1)
        return x


# Input with N * filters * (6,7)
# Output with N * 7
class PolicyHead():
    def __init__(self, weight_decay: float):
        self.conv1 = Conv2D(filters=2,
                            kernel_size=1,
                            # data_format='channels_first',
                            kernel_regularizer=l2(weight_decay))
        self.batch_norm = BatchNormalization(axis=3)
        self.relu = Activation('relu')
        self.fc1 = Dense(net_info.width, kernel_regularizer=l2(weight_decay))

    def build(self, input_: Input):
        x = self.conv1(input_)
        x = self.batch_norm(x)
        x = self.relu(x)
        # x = x.view(x.shape[0], 1, -1)
        x = Reshape((2 * net_info.area,))(x)
        x = self.fc1(x)
        # x = x.view(-1, net_info.width)
        return x


# Used in 8-ply testing
class ValueNet():
    def __init__(self, weight_decay: float):
        self.body = Body(weight_decay)
        self.value_head = ValueHead(weight_decay)

    def build(self, input_: Input):
        x = self.body.build(input_)
        x = self.value_head.build(x)
        return x

# value_net has no need of 'player to move' channel as it is only tested on 8ply boards
# input_ = Input((2, info.height, info.width), dtype='float32')
input_ = Input((info.height, info.width, 2), dtype='float32')
value_net = ValueNet(0.0001).build(input_)


# One can subclass their own models but better to avoid?
# https://keras.io/models/about-keras-models/
class Net():
    def __init__(self, weight_decay: float):
        self.body = Body(weight_decay)
        self.value_head = ValueHead(weight_decay)
        self.policy_head = PolicyHead(weight_decay)

    def build(self, input_: Input):
        # x = x.view(-1, net_info.channels, info.height, info.width)
        body_out = self.body.build(input_)
        value = self.value_head.build(body_out)
        policy = self.policy_head.build(body_out)

        model = Model(inputs=[input_], outputs=[value, policy])
        return model


class ModelWrapper():
    def __init__(self,
                 config: ModelConfig,
                 checkpoint: Optional[str] = None):
        self.config = config

        # input_ = Input((net_info.channels, info.height, info.width), dtype='float32')
        input_ = Input((info.height, info.width, net_info.channels), dtype='float32')

        self.model = Net(config.weight_decay).build(input_)
        # sth about a model creation

        # FIXME: https://keras.io/optimizers/ recommends RMS prop
        # I also quote "Much like Adam is essentially RMSprop with momentum, Nadam is Adam RMSprop with Nesterov momentum."
        self.optimiser = SGD(lr=config.initial_lr,
                             momentum=config.momentum,
                             # FIXME: hack
                             decay=config.gamma / config.milestones[0])
        self.model.compile(optimizer=self.optimiser,
                           loss=['mean_squared_error', 'categorical_crossentropy'],
                           # FIXME: investigate
                           metrics=None)

        # NOTE: can pass 8 ply as additional validation?!

        if checkpoint is not None:
            self.model.load_model(checkpoint)

        #FIXME: initializers for the layers?

        print(self.model.summary())

    def __call__(self, board: Board):
        board_array = np.expand_dims(board.to_array(), 0)
        board_array = np.moveaxis(board_array, 1, -1)
        value, prior = self.model.predict(board_array)
        return value.flatten(), prior.flatten()

    def save(self, file_name: str):
        self.model.save(file_name)

    def train(self,
              data: Sequence,
              n_epochs: int):
        #FIXME: check batch_size
        self.model.fit(self.create_sequence(*data), epochs=n_epochs)

    def evaluate_value_only(self, data: Sequence):
        stats = Stats()
        true_values = data.get_all_values()
        values, _ = self.model.predict_generator(data)
        # Tensorflow reverses model output, predition to my notation. But is that meaningless?!
        loss = losses.mean_squared_error(values, true_values)
        # FIXME: share the stats
        stats.update(values, true_values, loss)
        return stats

    def create_sequence(self, boards, values, policies):
        boards = np.moveaxis(boards, 1, -1)
        return Connect4Sequence(boards, values, policies, self.config.batch_size)


class Connect4Sequence(Sequence):
    def __init__(self, boards, values, policies, batch_size: int):
        assert len(boards) == len(values) == len(policies)
        self.boards = np.array(boards)
        self.values = np.array(values)
        self.policies = np.array(policies)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.values) / float(self.batch_size)))

    def __getitem__(self, idx):
        return np.array([self.boards[idx * self.batch_size:(idx + 1) * self.batch_size],
                         self.values[idx * self.batch_size:(idx + 1) * self.batch_size],
                         self.policies[idx * self.batch_size:(idx + 1) * self.batch_size]])

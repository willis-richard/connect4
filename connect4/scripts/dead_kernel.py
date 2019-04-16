import numpy as np
import pickle

np.random.seed(0)
from tensorflow.keras import optimizers

from sklearn.model_selection import train_test_split

n_epocs = 10000
epochs_per_stats = 1
batch_size = 2
test_size = 0.2
learning_rate = 0.002 * (batch_size / 1024.0)
momentum = 0.0

WORKING_DIR = '/home/richard/Downloads/nn/PSU_back/tf'


import torch
boards = torch.load('/home/richard/Downloads/connect4_boards.pth').numpy()
values = torch.load('/home/richard/Downloads/connect4_values.pth').numpy()

# Here we don't want to have the player to move channel
boards = boards[:, 1:]

board_train, board_test, value_train, value_test = train_test_split(boards, values, test_size=test_size, shuffle=True)


from connect4.neural.nn_tf import Connect4Sequence

# Also I think the shuffle is applied elsewhere
train_gen = Connect4Sequence(batch_size, board_train, value_train)
test_gen = Connect4Sequence(batch_size, board_test, value_test)



from connect4.neural.nn_tf import value_net as net

net.compile(
    optimizer=optimizers.Adam(),
    loss=['mean_squared_error'],
    # FIXME: investigate
    metrics=['accuracy'])

net.evaluate_generator(test_gen)

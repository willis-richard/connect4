from connect4.board import Board

from connect4.neural.config import ModelConfig
from connect4.neural.nn_pytorch import ModelWrapper

import numpy as np
import sys


if __name__ == "__main__":
    array = np.genfromtxt(sys.argv[1], dtype='c')
    o_pieces = np.zeros(array.shape, dtype=np.bool_)
    x_pieces = np.zeros(array.shape, dtype=np.bool_)
    o_pieces[np.where(array == b'o')] = 1
    x_pieces[np.where(array == b'x')] = 1

    print(o_pieces)
    print(x_pieces)
    board = Board(o_pieces, x_pieces)

    model_config = ModelConfig()
    model = ModelWrapper(model_config, file_name=sys.argv[2])
    print('{} value {}'.format(board, model(board)))

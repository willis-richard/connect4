from oinkoink.board import Board
from oinkoink.utils import Connect4Stats as info

from copy import deepcopy
import numpy as np
from typing import Callable, Dict, Optional, Tuple


class Evaluator():
    def __init__(self,
                 evaluate_fn: Callable,
                 position_table: Optional[Dict[Tuple, Tuple]] = None,
                 store_position: Optional[bool] = True):
        self.evaluate_fn = evaluate_fn
        self.position_table = {} if position_table is None else position_table
        self.store_position = store_position

    def __call__(self, board: Board):
        b_value = board.to_int_tuple()
        position_eval = self.position_table.get(b_value)
        if position_eval is None:
            position_eval = self.evaluate_fn(board)
            if self.store_position:
                self.position_table[b_value] = position_eval
        return deepcopy(position_eval)


def evaluate_centre(board: Board):
    value = 0.5 + \
        (np.einsum('ij,ij', board.o_pieces, value_grid)
         - np.einsum('ij,ij', board.x_pieces, value_grid)) \
        / float(value_grid_sum)
    return value


def evaluate_centre_with_prior(board: Board):
    value = evaluate_centre(board)
    return value, prior


def evaluate_nn(board: Board,
                model):
    value, prior = model(board)
    return float(value), prior


# Helper functions
centre = (info.height / 2.0, info.width / 2.0)

value_grid = np.stack([
    np.concatenate(
        (np.arange(centre[1]),
         np.flip(np.arange(int(centre[1])))))
    for i in range(info.height)])
value_grid = value_grid + np.transpose(np.stack([
    np.concatenate((np.arange(centre[0]),
                    np.flip(np.arange(int(centre[0])))))
    for i in range(info.width)]))

value_grid_t = np.transpose(value_grid)
value_grid_sum = np.sum(value_grid)

prior = np.ones((info.width,), dtype=float) / info.width

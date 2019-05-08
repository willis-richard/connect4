from connect4.board import Board
from connect4.utils import Connect4Stats as info

from copy import copy
import numpy as np
from typing import Callable, Dict, List, Optional, Set


class Evaluator():
    def __init__(self,
                 evaluate_fn: Callable,
                 position_table: Optional[Dict] = None,
                 result_table: Optional[Dict] = None,
                 store_position: Optional[bool] = True):
        self.evaluate_fn = evaluate_fn
        self.position_table = position_table if position_table is not None else {}
        self.result_table = result_table if result_table is not None else {}
        self.store_position = store_position

    def __call__(self, board: Board):
        b_value = board.to_int_tuple()
        position_eval = self.position_table.get(b_value)
        if position_eval is None:
            position_eval = self.evaluate_fn(board)
            if self.store_position:
                self.position_table[b_value] = position_eval
        return position_eval


def evaluate_centre(board: Board):
    value = 0.5 + \
        (np.einsum('ij,ij', board.o_pieces, info.value_grid)
         - np.einsum('ij,ij', board.x_pieces, info.value_grid)) \
        / float(info.value_grid_sum)
    return value


def evaluate_centre_with_prior(board: Board):
    value = evaluate_centre(board)
    prior = copy(info.prior)
    return value, prior


# FIXME: This is temporary until I have the net outputting priors
def evaluate_nn(board: Board,
                model):
    value, prior = model(board)
    # prior = softmax(prior)
    prior = copy(info.prior)
    return value, prior

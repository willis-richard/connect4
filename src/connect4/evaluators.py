from src.connect4.board import Board
from src.connect4.utils import Connect4Stats as info

from src.connect4.neural.network import Model

from copy import copy
from functools import partial
from typing import List, Set
import numpy as np


class Evaluator():
    def __init__(self, evaluate_fn):
        self.evaluate_fn = evaluate_fn
        self.position_table = {}
        self.result_table = {}

    def __call__(self, board: Board):
        if board in self.position_table:
            position_eval = self.position_table[board]
        else:
            position_eval = self.evaluate_fn(board)
            self.position_table[board] = position_eval
        return position_eval


class NetEvaluator(Evaluator):
    def __init__(self, evaluate_fn, model):
        self.model = model
        super().__init__(partial(evaluate_fn, model=self.model))


def evaluate_centre(board: Board):
    value = 0.5 + \
        (np.einsum('ij,ij', board.o_pieces, info.value_grid)
         - np.einsum('ij,ij', board.x_pieces, info.value_grid)) \
        / float(info.value_grid_sum)
    return value


def evaluate_centre_with_prior(board: Board):
    value = evaluate_centre(board)
    prior = copy(info.prior)
    prior = normalise_prior(board.valid_moves,
                            prior)
    return value, prior


def normalise_prior(valid_moves: Set, prior: List[float]):
    invalid_moves = set(range(info.width)).difference(valid_moves)
    if invalid_moves:
        for a in invalid_moves:
            prior[a] = 0.0
    prior = prior / np.sum(prior)
    return prior


def evaluate_nn(board: Board,
                model: Model):
    value, prior = model(board)
    value = value.cpu()
    value = value.view(-1)
    value = value.data.numpy()
    # prior = prior.cpu()
    # prior = prior.view(-1)
    # prior = prior.data.numpy()
    # prior = softmax(prior)
    prior = copy(info.prior)
    prior = normalise_prior(board.valid_moves, prior)
    return value, prior

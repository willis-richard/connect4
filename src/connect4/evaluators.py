from src.connect4.board import Board
from src.connect4.utils import Connect4Stats as info

from src.connect4.neural.network import Model

from anytree import Node
from functools import partial
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
    prior = info.policy_logits
    prior = normalise_prior(board.valid_moves,
                            prior)
    return value, prior


def normalise_prior(valid_moves, policy_logits):
    invalid_moves = set(range(info.width)).difference(valid_moves)
    if invalid_moves:
        for a in invalid_moves:
            policy_logits[a] = 0.0
    policy_logits = policy_logits / np.sum(policy_logits)
    return policy_logits


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
    prior = info.policy_logits
    prior = normalise_prior(board.valid_moves, prior)
    return value, prior

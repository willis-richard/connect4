from connect4.board import Board
from connect4.player import BasePlayer

import numpy as np


def training_game(player: BasePlayer):
    board = Board()
    boards = []
    moves = []
    values = []
    policies = []
    while board.result is None:
        move, value, tree = player.make_move(board)
        policy = tree.get_policy_max()

        boards.append(board.to_array())
        moves.append(move)
        values.append(value)
        policies.append(policy)

    values = create_values(values, board.result)

    return board.result, boards, moves, values, policies


def create_values(mcts_values, result):
    # FIXME: TD(lambda) algorithm?
    merged_values = (np.array(mcts_values, dtype='float') + result.value) / 2.0
    return merged_values
    # return np.array(mcts_values, dtype='float')

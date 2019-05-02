from connect4.board import Board
from connect4.player import BasePlayer

import numpy as np


def training_game(player: BasePlayer):
    board = Board()
    boards = []
    values = []
    policies = []
    history = []
    while board.result is None:
        move, value, tree = player.make_move(board)

        boards.append(board.to_array())
        values.append(value)
        policy = tree.get_policy_max()
        policies.append(policy)

        history.append((move, value, policy))

    values = create_values(values, board.result)
    print("Game finished by {}".format(player.name))

    return board.result, history, (boards, values, policies)


def create_values(mcts_values, result):
    # FIXME: TD(lambda) algorithm?
    merged_values = (np.array(mcts_values, dtype='float') + result.value) / 2.0
    return merged_values
    # return np.array(mcts_values, dtype='float')

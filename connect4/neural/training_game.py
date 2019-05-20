from connect4.board import Board
from connect4.player import BasePlayer

from copy import copy
import numpy as np
from typing import List, Optional


def training_game(player: BasePlayer):
    board = Board()
    game_data = GameData()
    while board.result is None:
        move, value, tree = player.make_move(board)
        policy = tree.get_policy_all()
        game_data.add_move(copy(board), move, value, policy)

    game_data.create_values(board.result)

    return game_data


class TrainingData:
    def __init__(self,
                 boards: Optional[List] = None,
                 values: Optional[List] = None,
                 policies: Optional[List] = None):
        self.boards = [] if boards is None else boards
        self.values = [] if values is None else values
        self.policies = [] if policies is None else policies

    def add(self, other: 'TrainingData'):
        self.boards.extend(other.boards)
        self.values = np.concatenate((self.values, other.values), 0)
        self.policies.extend(other.policies)

    def __repr__(self):
        return str([(b, v, p) for b, v, p in zip(self.boards,
                                                 self.values,
                                                 self.policies)])


class GameData():
    def __init__(self):
        self.result = None
        self.game = []
        self.boards = []
        self.values = []
        self.policies = []

    def add_move(self, board, move, value, policy):
        self.game.append((move, value, policy))
        self.boards.append(board)
        self.values.append(value)
        self.policies.append(policy)

    def create_values(self, result):
        # FIXME: TD(lambda) algorithm?
        self.result = result
        # self.values = (np.array(self.values, dtype='float') + result.value) / 2.0
        self.values = np.ones(len(self.values), dtype='float') * result.value

    @property
    def data(self):
        assert self.result is not None
        return TrainingData(self.boards, self.values, self.policies)

    def __str__(self):
        return "Result: {}, Game: {}, Data: {}".format(self.result,
                                                       self.game,
                                                       self.data)

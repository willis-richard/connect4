from oinkoink.board import Board

from oinkoink.neural.config import ModelConfig
from oinkoink.neural.training_game import TrainingData, GameData

import os
import pickle
from typing import List, Tuple


class GameStorage():
    def save(self,
             games: List[GameData],
             folder_path: str):
        with open(folder_path + '/games.pkl', 'wb') as f:
            pickle.dump(games, f)
        self.last_game = games[-1]

    def last_game_str(self):
        return game_str(self.last_game.moves,
                        self.last_game.values,
                        self.last_game.priors)


def game_str(moves: List,
             values: List,
             policies: List):
    board = Board()
    out_str = str(board)
    for move, value, policy in zip(moves, values, policies):
        board.make_move(move)
        out_str += '\nMove: {}  Value: {} Policy: {}\n{}'.format(move,
                                                                 value,
                                                                 policy,
                                                                 board)
    return out_str

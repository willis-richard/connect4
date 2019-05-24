from connect4.board import Board

from connect4.neural.config import ModelConfig
from connect4.neural.training_game import TrainingData, GameData

import os
import pickle
from typing import List, Tuple


class GameStorage():
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.iteration = 0
        file_list = os.listdir(folder_path + '/games')
        if file_list:
            iterations = [int(f.split('.')[0]) for f in file_list]
            self.iteration = max(iterations)

    @property
    def file_name(self):
        return "{}/games/{}.pth".format(self.folder_path, self.iteration)

    def save(self, games: List[GameData]):
        self.iteration += 1
        with open(self.file_name, 'wb') as f:
            pickle.dump(games, f)
        self.last_game = games[-1]

    def last_game_str(self):
        return game_str(self.last_game.moves,
                        self.last_game.values,
                        self.last_game.priors)


class NetworkStorage():
    def __init__(self,
                 folder_path: str,
                 config: ModelConfig,
                 model_wrapper_type):
        self.folder_path = folder_path
        self.iteration = 0
        file_list = os.listdir(folder_path + '/net')
        if file_list:
            iterations = [int(f.split('.')[0]) for f in file_list]
            self.iteration = max(iterations)
            print("Loading Network saved in file {}".format(self.file_name))
            self.model = model_wrapper_type(config, self.file_name)
        else:
            self.model = model_wrapper_type(config)

    @property
    def file_name(self):
        return "{}/net/{}.pth".format(self.folder_path, self.iteration)

    def train(self, data: TrainingData):
        self.model.train(data)
        self.iteration += 1
        self.model.save(self.file_name)

    def get_model(self):
        return self.model


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

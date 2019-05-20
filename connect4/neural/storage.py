from connect4.board import Board

from connect4.neural.config import ModelConfig
from connect4.neural.training_game import TrainingData

import os
import pickle
from typing import List, Tuple


class GameStorage():
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.iteration = 0
        file_list = os.listdir(folder_path)
        if file_list:
            iterations = [int(f.split('.')[1]) for f in file_list]
            self.iteration = max(iterations)
        self.games = []

    @property
    def file_name(self):
        return self.folder_path + '/games.' + str(self.iteration) + '.pkl'

    def save(self):
        self.iteration += 1
        with open(self.file_name, 'wb') as f:
            pickle.dump(self.games, f)
        self.games = []

    def save_game(self, game: List[Tuple[int, float]]):
        self.games.append(game)

    def save_games(self, games: List[List[Tuple[int, float]]]):
        self.games.extend(games)

    def last_game_str(self):
        return game_str(self.games[-1])


class NetworkStorage():
    def __init__(self,
                 folder_path: str,
                 config: ModelConfig,
                 model_wrapper_type):
        self.folder_path = folder_path
        self.iteration = 0
        file_list = os.listdir(folder_path)
        if file_list:
            iterations = [int(f.split('.')[1]) for f in file_list]
            self.iteration = max(iterations)
            print("Loading Network saved in file {}".format(self.file_name))
            self.model = model_wrapper_type(config, self.file_name)
        else:
            self.model = model_wrapper_type(config)

    @property
    def file_name(self):
        return self.folder_path + '/net.' + str(self.iteration) + '.pth'

    def train(self, data: TrainingData):
        self.model.train(data)
        self.iteration += 1
        self.model.save(self.file_name)

    def get_model(self):
        return self.model


def game_str(game: List):
    board = Board()
    out_str = str(board)
    for move, value, policy in game:
        board.make_move(move)
        out_str += '\nMove: {}  Value: {} Policy: {}\n{}'.format(move,
                                                                 value,
                                                                 policy,
                                                                 board)
    return out_str

from connect4.board import Board

from connect4.neural.config import ModelConfig
from connect4.neural.training_game import TrainingData

import numpy as np
import os
import pickle
from typing import List, Sequence, Tuple


class GameStorage():
    def __init__(self, folder_path: str):
        self.file_name = folder_path + '/games.pkl'
        self.games = []

    def save(self):
        if os.path.exists(self.file_name):
            with open(self.file_name, 'rb') as f:
                old_games = pickle.load(f)
            self.games = old_games + self.games
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
        self.save_model(self.model)
        self.iteration += 1

    def save_model(self, model):
        self.model = model
        self.model.save(self.file_name)

    def get_model(self):
        return self.model


def game_str(game: List):
    board = Board()
    out_str = str(board)
    for move, value in game:
        out_str += '\nMove: ' + str(move) + '  Value: ' + str(value)
        board.make_move(move)
        out_str += '\n' + str(board)

        return out_str

from connect4.neural.config import AlphaZeroConfig, ModelConfig
from connect4.neural.network import Model

import os
import pickle
from typing import List
import torch
from torch.utils.data import DataLoader, Dataset


class GameStorage():
    def __init__(self, folder_path: str):
        self.filename = folder_path + '/games.pkl'
        self.games = []

    def save(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                old_games = pickle.load(self.filename)
            self.games = old_games + self.games
        pickle.dumps(self.filename)
        self.games = []

    def save_game(self, game: List):
        self.games.append(game)

    def last_game_str(self):
        return game_str(self.games[-1])


class NetworkStorage():
    def __init__(self,
                 folder_path: str,
                 config: ModelConfig):
        self.folder_path = folder_path
        self.iteration = 0
        file_list = os.listdir(folder_path)
        if file_list:
            iterations = [int(f.split('.')[1]) for f in file_list]
            self.iteration = max(iterations)
            print(file_list, self.file_name)
            checkpoint = torch.load(self.file_name)
            self.model = Model(config, checkpoint)
        else:
            self.model = Model(config)

    @property
    def file_name(self):
        return self.folder_path + '/net.' + str(self.iteration) + '.pth'

    def train(self,
              data: DataLoader,
              n_epochs: int):
        self.model.train(data, n_epochs)
        self.save_model(self.model)
        self.iteration += 1

    def save_model(self, model):
        self.model = model
        torch.save(
            {
                'net_state_dict': self.model.net.state_dict(),
                'optimiser_state_dict': self.model.optimiser.state_dict(),
                'scheduler_state_dict': self.model.scheduler.state_dict()
            },
            self.file_name)

    def get_model(self):
        return self.model


class ReplayStorage():
    def __init__(self, config: AlphaZeroConfig):
        self.batch_size = config.batch_size
        # FIXME: not used
        # self.window_size = config.window_size
        self.reset()

    def reset(self):
        self.board_buffer = torch.Tensor()
        self.value_buffer = torch.Tensor()
        self.policy_buffer = torch.LongTensor()

    def save_game(self, boards, values, policies):
        self.board_buffer = torch.cat((self.board_buffer, boards), 0)
        self.value_buffer = torch.cat((self.value_buffer, values), 0)
        self.policy_buffer = torch.cat((self.policy_buffer, policies), 0)

    def get_data(self):
        data = Connect4Dataset(self.board_buffer,
                               self.value_buffer,
                               self.policy_buffer)
        return DataLoader(data, batch_size=self.batch_size, shuffle=True)

    # FIXME: Not used
    def sample_batch(self):
        import numpy
        # Sample uniformly across positions.
        move_sum = float(sum(len(g.history) for g in self.buffer))
        games = numpy.random.choice(
            self.buffer,
            size=self.batch_size,
            p=[len(g.history) / move_sum for g in self.buffer])
        game_pos = [(g, numpy.random.randint(len(g.history))) for g in games]
        return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]


class Connect4Dataset(Dataset):
    def __init__(self, boards, values, policies):
        assert len(boards) == len(values) == len(policies)
        self.boards = boards
        self.values = values
        self.policies = policies

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx: int):
        return (self.boards[idx],
                self.values[idx],
                self.policies[idx])


def game_str(game_history: List):
        board = Board()
        out_str = str(board)
        for move, value in game:
            out_str += '\nMove: ' + str(move) + '  Value: ' + str(value)
            board.make_move(move)
            out_str += '\n' + str(board)

        return out_str

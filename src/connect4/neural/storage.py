from src.connect4.neural.config import AlphaZeroConfig
from src.connect4.neural.network import Model

import torch
import os

from torch.utils.data import DataLoader, Dataset


class NetworkStorage():
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.iteration = 0
        file_list = os.listdir(folder_path)
        if file_list:
            latest_file = file_list[-1]
            print(file_list, latest_file)
            self.iteration = int(latest_file.split('.')[1])
            checkpoint = torch.load(self.file_name)
            self.model = Model(checkpoint)
        else:
            self.model = Model()

    @property
    def file_name(self):
        return self.folder_path + '.' + str(self.iteration) + '.pth'

    def train(self,
              data: DataLoader,
              n_epochs: int):
        self.model.train(data, n_epochs)
        self.save_model(self.model)

    def save_model(self, model):
        self.model = model
        torch.save(
            {
                'net_state_dict': self.model.net.state_dict(),
                'optimiser_state_dict': self.model.optimiser.state_dict()
            },
            self.file_name)

    def get_model(self):
        return self.model


class ReplayStorage():
    def __init__(self,
                 config: AlphaZeroConfig,
                 folder_path: str):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
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

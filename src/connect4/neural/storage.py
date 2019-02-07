from src.connect4.neural.config import AlphaZeroConfig
from src.connect4.neural.network import Net

import torch
import torch.optim as optim
import os


class NetworkStorage():
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.net = Net
        self.optimiser = optim.Adam(self.net.parameters())
        self.iteration = 0
        file_list = os.listdir()
        if file_list:
            latest_file = file_list[-1]
            self.iteration = int(latest_file.split('.')[1])
            checkpoint = torch.load(self.file_name)
            self.net.load_state_dict(checkpoint['net_state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    @property
    def file_name(self):
        return self.folder_path + '.' + str(self.iteration) + '.pth'

    def save_net(self, net, optimiser):
        self.net = net
        self.optimiser = optimiser
        torch.save(
            {
                'net_state_dict': self.net.state_dict(),
                'optimiser_state_dict': self.optimiser.state_dict()
            },
            self.file_name)

    def get_net(self):
        return self.net, self.optimiser

class ReplayStorage():
    def __init__(self,
                 config: AlphaZeroConfig,
                 folder_path: str):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = torch.Tensor()

    def save_game(self, game):
        self.buffer = torch.cat((self.buffer, game), 0)
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[:self.window_size]

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

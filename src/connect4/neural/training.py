from src.connect4.board import Board
from src.connect4.player import ComputerPlayer
from src.connect4.searching import MCTS
from src.connect4.utils import Result

from src.connect4.neural.config import AlphaZeroConfig
from src.connect4.neural.storage import NetworkStorage, ReplayStorage

import copy
import torch


class TrainingGame():
    def __init__(self,
                 player: ComputerPlayer,
                 board: Board,
                 replay_storage: ReplayStorage):
        self.player = player
        self.board = board
        self.replay_storage = replay_storage

    def play(self):
        boards = []
        policies = []
        while self.board.result is None:
            self.player.make_move(self.board)

            boards.append(self.board.to_tensor())
            policy = self.player.tree.get_policy()
            policies.append(torch.tensor(policy))

        boards = torch.stack(boards).squeeze()
        values = self.create_values(len(boards))
        policies = torch.stack(policies)
        # print(boards)
        # print(values)
        # print(policies)
        # game = torch.cat((boards,
        #                   values,
        #                   policies), dim=1)
        self.replay_storage.save_game(boards,
                                      values,
                                      policies)
        return self.board.result

    def create_values(self, length):
        # label board data with result
        values = torch.ones((length,), dtype=torch.float)
        if self.board.result == Result.o_win:
            values[1::2] = 0.
        elif self.board.result == Result.x_win:
            values[0::2] = 0.
        else:
            values *= 0.5
        return values


class TrainingLoop():
    def __init__(self,
                 config: AlphaZeroConfig,
                 folder_path: str):
        self.config = config
        self.nn_storage = NetworkStorage(folder_path)
        self.replay_storage = ReplayStorage(config, folder_path)

    def run(self):
        while True:
            self.loop()

    def loop(self):
        transition_t = {}
        player = ComputerPlayer('AlphaZero',
                                MCTS(MCTS.Config(simulations=100,
                                                 cpuct=9999)),
                                transition_t,
                                self.nn_storage.get_model())
        self.replay_storage.reset()

        for _ in range(self.config.n_training_games):
            TrainingGame(copy.copy(player),
                         Board(),
                         self.replay_storage).play()

        self.nn_storage.train(self.replay_storage.get_data(),
                              self.config.n_training_epochs)

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
                 replay_storage):
        self.player = player
        self.board = board
        self.replay_storage = replay_storage

    def play(self):
        boards = torch.Tensor()
        policies = torch.Tensor()
        while self.board.result is None:
            move, value, policy = self.player.makemove(self._board)
            torch.concatenate(boards, self._board.to_tensor())
            torch.concatenate(policies, policy)

        values = self.create_values()
        self.replay_storage.push(boards, values, policies)
        return self.board.result

    def create_values(self):
        # label board data with result
        values = torch.ones([2, 4], dtype=torch.float64)
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
                 nn_storage: NetworkStorage,
                 replay_storage: ReplayStorage):
        self.config = config
        self.nn_storage = nn_storage
        self.replay_storage = replay_storage

    def run(self):
        while True:
            self.loop()

    def loop(self):
        player = ComputerPlayer('AlphaZero',
                                MCTS(MCTS.Config(simulations=2500,
                                                 cpuct=9999)),
                                self.nn_storage.get_net())

        for _ in range(self.config.n_training_games):
            TrainingGame(copy.copy(player),
                         Board(),
                         self.nn_storage).play()

        self.nn_storage.train(self.config.n_training_epochs)

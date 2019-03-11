from src.connect4.board import Board
from src.connect4.game import Game
from src.connect4.player import BasePlayer

from copy import copy
import numpy as np
from typing import Set


def top_level_defined_play(x):
    return x.play().value


class Match():
    def __init__(self,
                 display: bool,
                 player_1: BasePlayer,
                 player_2: BasePlayer,
                 plies: int = 0,
                 switch: bool = False):
        self._player_1 = player_1
        self._player_2 = player_2

        ips = self.make_random_ips(plies)

        self.games = [Game(display,
                           copy(player_1),
                           copy(player_2),
                           copy(board))
                      for board in ips]

        self.n = len(self.games)

        if switch:
            self.games += \
                     [Game(display,
                           copy(player_2),
                           copy(player_1),
                           board)
                      for board in ips]
        self.switch = switch

    def play(self, agents=1):
        if agents == 1:
            results = []
            for i in range(len(self.games)):
                results.append(self.games[i].play().value)
        else:
            results = self.play_parallel(agents)

        results = np.array(results, dtype='f')
        # flip the results of the games where player_2 moved first
        if self.switch:
            results[self.n:] *= -1.0
            results[self.n:] += 1.0
        print("The results for " + self._player_1.name + " vs " + self._player_2.name + " are: {} wins, {} draws, {} losses"
              .format(np.sum(results == 1),
                      np.sum(results == 0.5),
                      np.sum(results == 0)))

    def play_parallel(self, agents):
        from multiprocessing import Pool

        with Pool(processes=agents) as pool:
            # results = pool.map(lambda x: x.play(), self.games)
            results = pool.map(top_level_defined_play, self.games)

        return results

    def make_random_ips(self, plies):
        ips = set()
        board = Board()
        self.expand(ips, board, plies)
        return ips

    def expand(self,
               ips: Set,
               board: Board,
               plies: int) -> None:
        if plies == 0:
            ips.add(board)
            return
        for move in board.valid_moves:
            new_board = copy(board)
            new_board.make_move(move)
            self.expand(ips, new_board, plies - 1)

        return

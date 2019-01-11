from src.connect4 import board
from src.connect4 import game

import copy
import numpy as np


def top_level_defined_play(x):
    return x.play()


class Match():
    def __init__(self,
                 display,
                 number_of_games,
                 player_1,
                 player_2):
        self.n = number_of_games
        self._player_1 = player_1
        self._player_2 = player_2

        # FIXME: play both sides
        # self.games = np.array([game.Game(display,
        #                                 copy.deepcopy(player_1),
        #                                 copy.deepcopy(player_2),
        #                                 board.Board())
        #                       for i in range(self.n)])
        self.games = [game.Game(display,
                                copy.deepcopy(player_1),
                                copy.deepcopy(player_2),
                                board.Board())
                      for i in range(self.n)]

    def play(self, agents=1):
        if agents == 1:
            results = np.empty_like(self.games, dtype='f')
            for i in range(self.n):
                results[i] = self.games[i].play().value
            print("The results are:\nPlayer one: {} wins, {} draws, {} losses"
                  .format(np.sum(results == 1),
                          np.sum(results == 0.5),
                          np.sum(results == 0)))
        else:
            results = self.play_parallel(agents)
            print(results)

    def play_parallel(self, agents):
        from multiprocessing import Pool

        with Pool(processes=agents) as pool:
            # results = pool.map(lambda x: x.play(), self.games)
            results = pool.map(top_level_defined_play, self.games)

        return results

    def make_random_ips(self, ply):
        for i in range(ply):
            print("nah")
            # moves = self._board.valid_moves
            # move = np.random.choice(moves)
            # self._board.make_move(move)

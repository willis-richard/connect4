import copy
import numpy as np

from src.connect4 import board
from src.connect4 import game


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
        #self.games = np.array([game.Game(display,
        #                                 copy.deepcopy(player_1),
        #                                 copy.deepcopy(player_2),
        #                                 board.Board())
        #                       for i in range(self.n)])
        self.games = [game.Game(display,
                                copy.deepcopy(player_1),
                                copy.deepcopy(player_2),
                                board.Board())
                      for i in range(self.n)]

    def play(self):
        results = np.empty_like(self.games, dtype='i')
        for i in range(self.n):
            results[i] = self.games[i].play()

        print("The results are:\nPlayer one: {} wins, {} draws and {} losses".format(np.sum(results == 1), np.sum(results == 0), np.sum(results == -1)))

    def play_parallel(self, agents):
        from multiprocessing import Pool

        with Pool(processes=agents) as pool:
            #results = pool.map(lambda x: x.play(), self.games)
            results = pool.map(top_level_defined_play, self.games)

        print(results)
        #print("The results are:\nPlayer one: {} wins, {} draws and {} losses".format(np.sum(results == 1), np.sum(results == 0), np.sum(results == -1)))

    def make_random_ips(self, ply):
        for i in range(ply):
              print("nah")
            #moves = self._board.get_valid_moves()
            #move = np.random.choice(moves)
            #self._board.make_move(move)

import copy
import numpy as np

from src.connect4 import board
from src.connect4 import game


class Match():
    def __init__(self,
                 number_of_games,
                 player_1,
                 player_2,
                 height=6,
                 width=7,
                 win_length=4):
        self.n = number_of_games
        self._player_1 = player_1
        self._player_2 = player_2

        # Check inputs are sane
        assert width >= win_length and height >= win_length
        # my has function converts to a u64
        assert height * width <= 64

        self.hash_value = np.array([2**x for x in range(height * width)])

        # FIXME: play both sides
        self.games = np.array([game.Game(copy.deepcopy(player_1),
                                         copy.deepcopy(player_2),
                                         board.Board(height,
                                                     width,
                                                     win_length,
                                                     self.hash_value))
                               for i in range(self.n)])

    def play(self):
        results = np.empty_like(self.games, dtype='i')
        for i in range(self.n):
            results[i] = self.games[i].play()

        print("The results are:\nPlayer one: {} wins, {} draws and {} losses".format(np.sum(results == 1), np.sum(results == 0), np.sum(results == -1)))

    def make_random_ips(self, ply):
        for i in range(ply):
              print("nah")
            #moves = self._board.get_valid_moves()
            #move = np.random.choice(moves)
            #self._board.make_move(move)

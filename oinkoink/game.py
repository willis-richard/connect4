from oinkoink.board import Board
from oinkoink.player import BasePlayer
from oinkoink.utils import Side

import numpy as np


class Game():
    def __init__(self,
                 display: bool,
                 player_o: BasePlayer,
                 player_x: BasePlayer,
                 board: Board):
        self.display = display
        self._player_o = player_o
        self._player_x = player_x
        self._board = board
        self.move_history = np.empty((0,), dtype='uint8')

    def play(self):
        if self.display:
            print("Game between", self._player_o, " and ", self._player_x)
            print(self._board)
        while self._board.result is None:
            if self._board.player_to_move == Side.o:
                move, value, tree = self._player_o.make_move(self._board)
                name = self._player_o.name
            else:
                move, value, tree = self._player_x.make_move(self._board)
                name = self._player_x.name
            if self.display:
                if tree is None:
                    print("{} selected move: {}".format(name, move))
                else:
                    prior = tree.get_visit_count_policy()
                    print("{} selected move: {}, value: {}, prior: {}".
                          format(name, move, value, prior))
                print(self._board)
            self.move_history = np.append(self.move_history, move)
        return self._board.result

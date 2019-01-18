from src.connect4.utils import Side

import numpy as np


class Game():
    def __init__(self,
                 display,
                 player_o,
                 player_x,
                 board_):
        self.display = display
        self._player_o = player_o
        self._player_x = player_x
        self._board = board_
        self.move_history = np.empty((0,), dtype='uint8')

    def play(self):
        if self.display:
            print("Game between", self._player_o, " and ", self._player_x)
            print(self._board)
        while self._board.result is None:
            if self._board._player_to_move == Side.o:
                move = self._player_o.make_move(self._board)
            else:
                move = self._player_x.make_move(self._board)
            self.move_history = np.append(self.move_history, move)
            if self.display:
                print(self._board)
        return self._board.result

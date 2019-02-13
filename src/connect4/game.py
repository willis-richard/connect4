from src.connect4.board import Board
from src.connect4.player import BasePlayer
from src.connect4.utils import Side

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
        while self._board.result is None:
            if self.display:
                print(self._board)
            if self._board._player_to_move == Side.o:
                move, _, _ = self._player_o.make_move(self._board)
                print(self._player_o.name + " selected move: ", move)
            else:
                move, _, _ = self._player_x.make_move(self._board)
                print(self._player_x.name + " selected move: ", move)
            self.move_history = np.append(self.move_history, move)
        return self._board.result

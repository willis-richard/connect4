from connect4 import board
from connect4 import player

import numpy as np


class Connect4():
    def __init__(self,
                 player_o_name,
                 player_x_name,
                 player_o_human=False,
                 player_x_human=False,
                 height=6,
                 width=7,
                 win_length=4):
        # Check inputs are sane
        assert width >= win_length and height >= win_length
        # my has function converts to a u64
        assert height * width <= 64

        self.hash_value = np.array([2**x for x in range(height * width)])

        self._board = board.Board(height,
                                  width,
                                  win_length,
                                  self.hash_value)

        self._player_o = player.HumanPlayer(player_o_name, 1, self._board) if player_o_human else player.ComputerPlayer(player_o_name, 1, self._board, 5)
        self._player_x = player.HumanPlayer(player_x_name, -1, self._board) if player_x_human else player.ComputerPlayer(player_x_name, -1, self._board, 5)

    def play(self):
        print("Match between", self._player_o, " and ", self._player_x)
        self._board.display()
        while self._board.result is None:
            if self._board.player_to_move == 'o':
                self._player_o.make_move()
            else:
                self._player_x.make_move()
            self._board.display()
            self._board.check_terminal_position()

        if self._board.result == 1:
            result = "o wins"
        elif self._board.result == -1:
            result = "x wins"
        else:
            result = "draw"

        print("The result is: ", result)

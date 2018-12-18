import numpy as np


class Game():
    def __init__(self,
                 player_o,
                 player_x,
                 board_):
        self._player_o = player_o
        self._player_x = player_x
        self._board = board_
        self.move_history = np.empty((0,), dtype='uint8')

    def play(self):
        print("Game between", self._player_o, " and ", self._player_x)
        self._board.display()
        while self._board.result is None:
            if self._board.player_to_move == 'o':
                move = self._player_o.make_move(self._board)
            else:
                move = self._player_x.make_move(self._board)
            self.move_history = np.append(self.move_history, move)
            self._board.display()
            self._board.check_terminal_position()
        return self._board.result

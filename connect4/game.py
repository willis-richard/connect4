from connect4 import board
from connect4 import player


class Connect4():
    def __init__(self, player_o_name, player_x_name, player_o_human=False, player_x_human=False):
        self._board = board.Board()
        self._player_o = player.HumanPlayer(player_o_name, 0, self._board) if player_o_human else player.ComputerPlayer(player_o_name, 0, self._board)
        self._player_x = player.HumanPlayer(player_x_name, 1, self._board) if player_x_human else player.ComputerPlayer(player_x_name, 1, self._board)

    def play(self):
        print("Match between", self._player_o, " and ", self._player_x)
        self._board.display()
        while not self._board.result:
            if self._board.player_to_move == 'o':
                self._player_o.make_move()
            else:
                self._player_x.make_move()
            self._board.display()

        if self._board.result == 1:
            result = "o wins"
        elif self._board.result == 2:
            result = "x wins"
        else:
            result = "draw"

        print("The result is: ", result)

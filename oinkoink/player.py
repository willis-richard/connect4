from oinkoink import tree
from oinkoink.utils import Side

from typing import Dict, Optional


class BasePlayer():
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "Player: " + self.name

    def make_move(self, board):
        raise NotImplementedError


class HumanPlayer(BasePlayer):
    def __init__(self, name):
        super().__init__(name)

    def make_move(self, board):
        move = -1
        while move not in board.valid_moves:
            try:
                move = int(input("Enter " + self.name + " (" +
                                 Side.as_str(board.player_to_move) + "'s) move:"))
            except ValueError:
                print("Not a valid move. Try again:")
                pass
        board.make_move(int(move))
        return move, None, None

    def __str__(self):
        return super().__str__() + ", type: Human"

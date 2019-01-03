from src.connect4 import tree

from src.connect4.utils import Connect4Stats as info

import numpy as np


class BasePlayer():
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "Player: " + self.name

    @property
    def side(self):
        return self._side

    @side.setter
    def side(self, side):
        assert side in [-1, 1]
        self._side = side


class HumanPlayer(BasePlayer):
    def __init__(self, name):
        super().__init__(name)

    def make_move(self, board):
        move = -1
        while move not in board.valid_moves:
            try:
                move = int(input("Enter " + self.name + " (" +
                                 board.player_to_move + "'s) move:"))
            except ValueError:
                print("Try again dipshit")
                pass
        board.make_move(int(move))
        return move

    def __str__(self):
        return super().__str__() + ", type: Human"


class ComputerPlayer(BasePlayer):
    def __init__(self, name, strategy):
        super().__init__(name)
        self.tree = tree.Connect4Tree(strategy.Evaluation,
                                      self.evaluate_position)
        self.search_fn = strategy.get_search_fn()

    def make_move(self, board):
        self.tree.update_root(board)
        move, value = self.search_fn(tree=self.tree,
                                     board=board,
                                     side=self.side)

        if value == self.side:
            print("Trash! I will crush you.")
        elif value == -1 * self.side:
            print("Ah fuck you lucky shit")

        print(self.name + " selected move: ", move)
        board.make_move(move)

        return move

    @staticmethod
    def evaluate_position(board):
        # return np.sum(np.multiply(board.o_pieces, info.value_grid) -
        # np.multiply(board.x_pieces, info.value_grid)) \
        # / float(info.value_grid_sum)
        return (np.einsum('ij,ij', board.o_pieces, info.value_grid)
                - np.einsum('ij,ij', board.x_pieces, info.value_grid)) \
                / float(info.value_grid_sum)

    def __str__(self):
        return super().__str__() + ", type: Computer"

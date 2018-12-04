from connect4 import tree

from anytree import RenderTree

import numpy as np


class BasePlayer():
    def __init__(self, name, side, board):
        self.name = name
        self.side = side
        self._board = board

    def __str__(self):
        return "Player: " + self.name


class HumanPlayer(BasePlayer):
    def __init__(self, name, side, board):
        super().__init__(name, side, board)

    def make_move(self):
        move = str(input("Enter player " + self._board.player_to_move + "'s move:"))
        while len(move) != 1 or int(move) not in self._board.valid_moves():
            print("Try again dipshit")
            move = str(input("Enter player " + self._board.player_to_move + "'s move:"))
        self._board.make_move(int(move))

    def __str__(self):
        return super().__str__() + ", type: Human"


class ComputerPlayer(BasePlayer):
    def __init__(self, name, side, board, depth):
        super().__init__(name, side, board)
        self.tree = tree.Connect4Tree(board)
        self.depth = depth

    def make_move(self):
        self.tree.update_root(self._board)
        self.tree.expand_node(self.tree.root, self.depth)
        self.tree.nega_max(self.tree.root, self.depth, self.side)

        moves = {}
        for node in self.tree.root.children:
            move = node.name[0]
            moves[move] = node.data.node_eval.get_value()

        print("POSSIBLE MOVES: ", moves)
        # FIXME: same selection problem
        if self.side == 1:
            max = -2
            best_moves = []
            for move, value in moves.items():
                if value > max:
                    best_moves = [move]
                    max = value
                elif value == max:
                    best_moves.append(move)
        else:
            max = 2
            best_moves = []
            for move, value in moves.items():
                if value < max:
                    best_moves = [move]
                    max = value
                elif value == max:
                    best_moves.append(move)

        best_moves = np.array(best_moves)
        distance_to_middle = np.abs(best_moves - self._board._width / 2.0)
        idx = np.argsort(distance_to_middle)
        best_move, best_move_value = best_moves[idx[0]], moves[best_moves[idx[0]]]

        if best_move_value == self.side:
            print("Trash! I will crush you.")
        elif best_move_value == -1 * self.side:
            print("Ah fuck you lucky shit")

        print("Best move selected: ", best_move)
        self._board.make_move(best_move)
        return best_move # for testing

    def __str__(self):
        return super().__str__() + ", type: Computer"

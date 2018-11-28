from connect4 import tree

import copy

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
        print("FYI valid moves are: ", self._board.valid_moves())
        move = str(input("Enter player " + self._board.player_to_move + "'s move:"))
        while len(move) != 1 or int(move) not in self._board.valid_moves():
            print("Try again dipshit")
            move = str(input("Enter player " + self._board.player_to_move + "'s move:"))
        self._board.make_move(int(move))

    def __str__(self):
        return super().__str__() + ", type: Human"



class ComputerPlayer(BasePlayer):
    def __init__(self, name, side, board):
        super().__init__(name, side, board)
        self.tree = tree.Connect4Tree(board)

    def make_move(self):
        self.tree.update_root(self._board)
        self.tree.nega_max(self.tree.root, 2)

        #self.tree.nega_max(self._board, 2)

        moves = {}
        for node in self.tree.root.children:
            move = node.name[0]
            moves[move] = node.data.node_eval.tree_value

        print("POSSIBLE MOVES: ", moves)
        max = -2 * self.side
        best_moves = []
        for move, value in moves.items():
            if value > max:
                best_moves = [move]
                max = value
            elif value == max:
                best_moves.append(move)

        best_moves = np.array(best_moves)
        distance_to_middle = np.abs(best_moves - self._board._width / 2.0)
        idx = np.argsort(distance_to_middle)
        best_move = moves[idx[0]]

        if best_move == self.side:
            print("Trash! I will crush you.")
        elif best_move == -1 * self.side:
            print("Ah fuck you lucky shit")
        else:
            print("zzz as if")

        self._board.make_move(best_move)

    def __str__(self):
        return super().__str__() + ", type: Computer"

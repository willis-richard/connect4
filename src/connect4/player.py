from src.connect4 import tree

import numpy as np


class BasePlayer():
    def __init__(self, name, side):
        self.name = name
        self.side = side

    def __str__(self):
        return "Player: " + self.name


class HumanPlayer(BasePlayer):
    def __init__(self, name, side):
        super().__init__(name, side)

    def make_move(self, board):
        move = -1
        while move not in board.valid_moves():
            try:
                move = int(input("Enter player " + board.player_to_move + "'s move:"))
            except ValueError:
                pass
            print("Try again dipshit")
        board.make_move(int(move))
        return move

    def __str__(self):
        return super().__str__() + ", type: Human"


class ComputerPlayer(BasePlayer):
    def __init__(self, name, side, depth):
        super().__init__(name, side)
        self.tree = tree.Connect4Tree()
        self.depth = depth

    def make_move(self, board):
        self.tree.update_root(board)
        self.tree.expand_node(self.tree.root, self.depth)
        self.tree.nega_max(self.tree.root, self.depth, self.side)

        moves = {}
        for node in self.tree.root.children:
            move = node.name
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
        distance_to_middle = np.abs(best_moves - board._width / 2.0)
        idx = np.argsort(distance_to_middle)
        best_move, best_move_value = best_moves[idx[0]], moves[best_moves[idx[0]]]

        if best_move_value == self.side:
            print("Trash! I will crush you.")
        elif best_move_value == -1 * self.side:
            print("Ah fuck you lucky shit")

        print("Best move selected: ", best_move)
        board.make_move(best_move)

        return best_move

    def __str__(self):
        return super().__str__() + ", type: Computer"

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
    def __init__(self, name, search_fn):
        super().__init__(name)
        self.tree = tree.Connect4Tree(self.evaluate_position)
        self.search_fn = search_fn
        # static member

    def make_move(self, board):
        self.tree.update_root(board)
        self.search_fn(tree=self.tree, board=board, side=self.side)

        moves = np.array([(n.name) for n in self.tree.root.children])
        values = np.array([(n.data.node_eval.get_value())
                           for n in self.tree.root.children])
        idx = np.argmax(values * self.side)
        # best_moves = moves[values == values[idx]]
        best_move_value = values[idx]

        best_move = moves[idx]
        # distance_to_middle = np.abs(best_moves - info.width / 2.0)
        # idx = np.argsort(distance_to_middle)
        # best_move = best_moves[idx[0]]

        if best_move_value == self.side:
            print("Trash! I will crush you.")
        elif best_move_value == -1 * self.side:
            print("Ah fuck you lucky shit")

        print(self.name + " selected move: ", best_move)
        board.make_move(best_move)

        return best_move

    @staticmethod
    def evaluate_position(board):
        # return np.sum(np.multiply(board.o_pieces, info.value_grid) -
        # np.multiply(board.x_pieces, info.value_grid)) \
        # / float(info.value_grid_sum)
        return (np.einsum('ij,ij', board.o_pieces, info.value_grid)
                - np.einsum('ij,ij', board.x_pieces, info.value_grid)) \
                / float(info.value_grid_sum)

    @staticmethod
    def grid_search(tree, board, side, depth):
        tree.expand_tree(tree.root, depth)
        tree.nega_max(tree.root, depth, side)

    @staticmethod
    def mcts(tree, board, side, simulations):
        def ucb_score(node, child):
            return 0

        def select_child(node):
            # guaranteed at least one non-terminally forcing child
            if not node.children:
                tree.expand_tree(node, 1)

            non_terminal_actions = set(node.data.valid_moves). \
                                   difference(node.data.terminal_children)
            non_terminal_children = [child for child in node.children if
                                     child.name in non_terminal_actions]
            _, child = max((ucb_score(node, child), child)
                           for child in non_terminal_children)

            return child

        for _ in range(simulations):
            node = tree.root
            while node.evaluated():
                node = select_child(node)

            if node.visited():
                tree.expand_tree(node, 1)
                action = select_action(node)
                node = tree.take_action(action, node)

            # while node.expandable():
            #     action = select_action(node)
            #     actions_explored = [c.name for c in node.children]
            #     if action in actions_explored:
            #         node = node.children[actions_explored == action]
            #     else:
            #         break

            # node = tree.take_action(action, node)

            node.evaluate()
            tree.backpropagage(node)

    def __str__(self):
        return super().__str__() + ", type: Computer"

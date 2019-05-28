from connect4.board_c import Board
from connect4.utils import Connect4Stats as info, Side, value_to_side

from anytree import Node
from copy import copy
import numpy as np
from scipy.special import softmax


# Give Node an __eq__ operator -> will be the same if the boards are the same
# def node_eq(self, other):
#     return self.data.board == other.data.board


# def node_gt(self, other):
#     return self.name > other.name


# def node_hash(self):
#     return hash(self.data.board)


# Node.__eq__ = node_eq
# Node.__gt__ = node_gt
# Node.__hash__ = node_hash


class BaseNodeData():
    """Node data is a pair of a board and some evaluation data"""
    def __init__(self,
                 board: Board):
        self.board = board
        self.valid_moves = board.valid_moves
        self.position_value = None
        self.search_value = None

    def value(self, side: Side):
        if self.board.result is not None:
            return value_to_side(self.board.result.value, side)
        elif self.search_value is not None:
            return value_to_side(float(self.search_value), side)
        elif self.position_value is not None:
            return value_to_side(float(self.position_value), side)
        else:
            # position is unknown - assume lost
            return 0.0

    def __str__(self):
        return \
            "board_result: " + str(self.board.result) + \
            ",  position_value: (" + str(self.position_value) + ")" + \
            ",  search_value: (" + str(self.search_value) + ")"

    def __repr__(self):
        return \
            "board: " + str(self.board) + \
            ",  valid_moves: " + str(self.valid_moves) + \
            ",  (board_result: " + str(self.board.result) + \
            ",  position_value: (" + str(self.position_value) + ")" + \
            ",  search_value: (" + str(self.search_value) + "))"


class Tree():
    def __init__(self,
                 board: Board,
                 node_data_type):
        self.side = board.player_to_move
        self.node_data_type = node_data_type
        self.root = self.create_node('root', copy(board))

    def get_node_value(self, node):
        return node.data.value(self.side)

    def select_best_move(self):
        value, move = max(((self.get_node_value(c),
                           c.name)
                          for c in self.root.children))

        absolute_value = value_to_side(value, self.side)

        return move, absolute_value

    def select_softmax_move(self):
        moves = []
        values = []
        for c in self.root.children:
            moves.append(c.name)
            values.append(self.get_node_value(c))
        values = softmax(values)

        idx = np.random.choice(range(len(moves)), p=values)
        absolute_value = value_to_side(values[idx], self.side)

        return moves[idx], absolute_value

    def get_policy(self):
        policy = np.zeros((info.width,))
        for c in self.root.children:
            policy[c.name] = self.get_node_value(c)
        policy_sum = np.sum(policy)
        if policy_sum == 0.0:
            for c in self.root.children:
                policy[c.name] = 1.0
            return policy / len(self.root.children)
        else:
            return policy / policy_sum

    def create_node(self, name, board, parent=None):
        node_data = self.node_data_type(board)

        return Node(name, parent=parent, data=node_data)

    def expand_node(self,
                    node: Node,
                    plies: int):
        if plies == 0 or node.data.board.result is not None:
            return

        if not node.children:
            for move in node.data.valid_moves:
                new_board = copy(node.data.board)
                new_board.make_move(move)
                child = self.create_node(move, new_board, parent=node)

        for child in node.children:
            self.expand_node(child, plies - 1)

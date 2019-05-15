from connect4.board import Board
from connect4.utils import Connect4Stats as info, Result, Side, value_to_side

from anytree import Node
from copy import copy
import numpy as np
from scipy.special import softmax
from typing import Dict, Optional, Tuple


class BaseNodeData():
    """Node data is a pair of a board and some evaluation data"""
    def __init__(self,
                 board: Board):
        self.board = board
        self.valid_moves = board.valid_moves
        self.position_value = None
        self.search_value = None

    def value(self, side: Side):
        if self.board.result:
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
        self.side = board._player_to_move
        self.node_data_type = node_data_type
        self.root = self.create_node('root', copy(board))

    def get_node_value(self, node, side: Optional[Side] = None):
        if side is None:
            return node.data.value(self.side)
        else:
            return node.data.value(side)

    def select_best_move(self):
        _, move, value = max(((self.get_node_value(c),
                               c.name,
                               self.get_node_value(c, Side.o))
                              for c in self.root.children))
        return move, value

    def select_softmax_move(self):
        moves = []
        values = []
        for c in self.root.children:
            moves.append(c.name)
            values.append(self.get_node_value(c))
        values = softmax(values)

        idx = np.random.choice(range(len(moves)), p=values)

        return moves[idx], values[idx]

    def get_policy_all(self):
        # If using that multiclass stuff
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

        node = Node(name, parent=parent, data=node_data)
        return node

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

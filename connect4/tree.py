from connect4.board import Board
from connect4.utils import Connect4Stats as info, Result, Side, value_to_side

from anytree import Node
from copy import copy
import numpy as np
from typing import Dict, Tuple


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
                 result_table: Dict[int, Result],
                 node_data_type):
        self.side = board._player_to_move
        self.result_table = result_table
        self.node_data_type = node_data_type
        self.root = self.create_node('root', copy(board))

    def get_node_value(self, node):
        return node.data.value(self.side)

    def select_best_move(self) -> Tuple[int, float]:
        value, action = max(((self.get_node_value(c), c.name)
                             for c in self.root.children))
        return action, value

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
        # If pytorch CrossEntropy
        # _, action = max((self.get_node_value(c), c.name)
        #                 for c in self.root.children)
        # return action

    def create_node(self, name, board, parent=None):
        if board in self.result_table:
            board_result = self.result_table[board]
            board.result = board_result
        else:
            board_result = board.check_terminal_position()
            self.result_table[board] = board_result
        # FIXME: copy required?
        node_data = self.node_data_type(copy(board))

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
                new_board._make_move(move)
                child = self.create_node(move, new_board, parent=node)

        for child in node.children:
            self.expand_node(child, plies - 1)

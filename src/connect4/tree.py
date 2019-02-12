from src.connect4.board import Board
from src.connect4.utils import Connect4Stats as info
from src.connect4.utils import Side, value_to_side

from anytree import Node
from copy import copy
import numpy as np
from typing import Dict, Tuple


class BaseNodeData():
    """Node data is a pair of a board and some evaluation data"""
    def __init__(self,
                 board: Board,
                 position_evaluation,
                 search_evaluation):
        self.board = board
        self.valid_moves = board.valid_moves
        self.position_evaluation = position_evaluation
        self.search_evaluation = search_evaluation

    def update_position_value(self, value):
        self.position_evaluation.update_value(value)

    def update_search_value(self, value):
        self.search_evaluation.update_value(value)

    @property
    def value(self):
        if self.board.result:
            return self.board.result.value
        elif self.search_evaluation.value is not None:
            return self.search_evaluation.value
        elif self.position_evaluation.value is not None:
            return self.position_evaluation.value
        else:
            # position is unknown - assume lost
            return 0

    def __repr__(self):
        return \
            "board: " + str(self.board) + \
            ",  valid_moves: " + str(self.valid_moves) + \
            ",  (board_result: " + str(self.board.result) + \
            ",  position_evaluation: (" + str(self.position_evaluation) + ")" + \
            ",  search_evaluation: (" + str(self.search_evaluation) + "))"


class Tree():
    def __init__(self,
                 node_data_type,
                 position_evaluation_type,
                 transition_t: Dict = None):
        self.node_data_type = node_data_type
        self.position_evaluation_type = position_evaluation_type

        # transition_t is a map from board to NodeEvaluation
        # Ideally it would be to NodeData, but I have not solved the
        # transposition problem in my search tree yet
        self.transition_t = dict() if transition_t is None else transition_t
        # FIXME: Actually only set when passed a board
        self.side = Side.o

    def update_root(self, board: Board):
        self.root = self.create_node('root', copy(board))
        # self.transition_t.age(board.age)
        self.side = board._player_to_move

    def get_node_value(self, node) -> float:
        return value_to_side(node.data.value, self.side)

    def select_best_move(self) -> Tuple[int, float]:
        value, action = max(((self.get_node_value(c), c.name)
                             for c in self.root.children))
        return action, value

    def get_policy(self):
        # policy = np.zeros((info.width,))
        # for c in self.root.children:
        #     policy[c.name] = c.data.search_evaluation.visit_count
        # policy = policy / np.sum(policy)
        # FIXME: was visit_count
        _, action = max((c.data.value, c.name)
                        for c in self.root.children)
        return action

    def create_node(self, name, board, parent=None):
        if board in self.transition_t:
            board_result, position_evaluation = self.transition_t[board]
            board.result = board_result
        else:
            board_result = board.check_terminal_position()
            position_evaluation = self.position_evaluation_type()
            self.transition_t[board] = (board_result, position_evaluation)
        node_data = self.node_data_type(board,
                                        position_evaluation)

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
                self.create_node(move, new_board, parent=node)

        for child in node.children:
            self.expand_node(child, plies - 1)

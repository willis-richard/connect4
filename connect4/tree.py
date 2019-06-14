from connect4.board_c import Board
from connect4.utils import Connect4Stats as info, Side, value_to_side

from anytree import Node
from copy import deepcopy
import numpy as np
from scipy.special import softmax


# Give Node an __eq__ operator -> will be the same if the boards are the same
# def node_eq(self, other):
#     return self.data.board == other.data.board


def node_gt(self, other):
    return self.name > other.name


# def node_hash(self):
#     return hash(self.data.board)


# Node.__eq__ = node_eq
Node.__gt__ = node_gt
# Node.__hash__ = node_hash


class NodeData():
    """Node data is a pair of a board and some evaluation data"""
    def __init__(self,
                 board: Board):
        self.board = board
        self.valid_moves = board.valid_moves
        self.position_value = None
        self.search_value = None

    @property
    def absolute_value(self):
        if self.board.result is not None:
            return self.board.result.value
        elif self.search_value is not None:
            return float(self.search_value)
        elif self.position_value is not None:
            return float(self.position_value)
        else:
            return None

    def value(self, side: Side):
        absolute_value = self.absolute_value
        if absolute_value is not None:
            return value_to_side(absolute_value, side)
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
    def __init__(self, board: Board):
        self.side = board.player_to_move
        self.root = self.create_node('root', deepcopy(board))

    def get_node_value(self, node):
        return node.data.value(self.side)

    def best_move(self):
        _, child = max(((self.get_node_value(child), child)
                       for child in self.root.children))

        return child

    def most_visited(self):
        _, child = max(((child.data.search_value.visit_count
                         if child.data.search_value is not None
                         else 0,
                         child)
                       for child in self.root.children))

        return child

    def softmax_visit_count(self):
        visit_counts = [c.data.search_value.visit_count
                        if c.data.search_value
                        else 0
                        for c in self.root.children]
        probabilties = softmax(visit_counts)

        idx = np.random.choice(range(len(visit_counts)), p=probabilties)

        return self.root.children[idx]

    def get_values_policy(self):
        policy = np.zeros((info.width,))
        for c in self.root.children:
            policy[c.name] = self.get_node_value(c)
        self.normalise_policy(policy)
        return policy

    def get_visit_count_policy(self):
        policy = np.zeros((info.width,))
        for c in self.root.children:
            if c.data.search_value is not None:
                policy[c.name] = c.data.search_value.visit_count
        return policy

    def normalise_policy(self, policy):
        """Operates in-place"""
        policy_sum = np.sum(policy)
        if policy_sum == 0.0:
            for c in self.root.children:
                policy[c.name] = 1.0
            policy /= len(self.root.children)
        else:
            policy /= policy_sum

    def create_node(self, name, board, parent=None):
        node_data = NodeData(board)

        return Node(name, parent=parent, data=node_data)

    def expand_node(self,
                    node: Node,
                    plies: int):
        if plies == 0 or node.data.board.result is not None:
            return

        if not node.children:
            for move in node.data.valid_moves:
                new_board = deepcopy(node.data.board)
                new_board.make_move(move)
                child = self.create_node(move, new_board, parent=node)

        for child in node.children:
            self.expand_node(child, plies - 1)

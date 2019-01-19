from src.connect4.board import Board
from src.connect4.utils import Side, result_to_side

from anytree import Node
from copy import copy


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
        if self.search_evaluation.value is not None:
            return self.search_evaluation.value
        elif self.position_evaluation.value is not None:
            return self.position_evaluation.value
        else:
            # position is unknown - assume lost
            return 0

    def set_child_map(self, children):
        self.child_map = {c.name: c for c in children}

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
                 evaluation_type):
        self.node_data_type = node_data_type
        self.evaluation_type = evaluation_type

        # transition_t is a map from board to NodeEvaluation
        # Ideally it would be to NodeData, but I have not solved the
        # transposition problem in my search tree yet
        self.transition_t = dict()

    def update_root(self, board):
        self.root = self.create_node('root', copy(board))
        # self.transition_t.age(board.age)

    def take_action(self, action, node):
        child_names = [c.name for c in node.children]
        if action in child_names:
            return node.children[child_names.index(action)]

        return self.create_child(action, node)

    def create_child(self, action, node):
        new_board = copy(node.data.board)
        new_board._make_move(action)
        return self.create_node(action, new_board, parent=node)

    def create_node(self, name, board, parent=None):
        if board in self.transition_t:
            board_result, node_evaluation = self.transition_t[board]
            board.result = board_result
        else:
            board_result = board.check_terminal_position()
            node_evaluation = self.evaluation_type()
            self.transition_t[board] = (board_result, node_evaluation)
        node_data = self.node_data_type(board,
                                        node_evaluation)

        node = Node(name, parent=parent, data=node_data)
        return node

    def expand_node(self,
                    node: Node,
                    plies: int):
        if plies == 0 or node.data.board.result is not None:
            return

        if not node.children:
            for move in node.data.valid_moves:
                self.create_child(move, node)
            node.data.set_child_map(node.children)

        for child in node.children:
            self.expand_node(child, plies - 1)

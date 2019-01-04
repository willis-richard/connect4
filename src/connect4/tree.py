import anytree
import copy


class NodeData():
    """Node data is a pair of a board and some evaluation data"""
    def __init__(self, board, evaluation):
        self.board = board
        self.valid_moves = board.valid_moves
        self.evaluation = evaluation

    def unexplored_moves(self, children):
        return self.valid_moves.difference([c.name for c in children])

    @property
    def value(self):
        if self.board.result is not None:
            return self.board.result
        return self.evaluation.value


class Connect4Tree():
    def __init__(self, evaluation_type, transition_t=None):
        self.evaluation_type = evaluation_type

        # transition_t is a map from board to NodeEvaluation
        self.transition_t = dict()
        # at least while aging is not a problem, a dict() is
        # faster than my other implementations
        # self.transition_t = TransitionTableDictOfDict() \
        #     if transition_t is None else transition_t

    def create_node(self, name, board, parent=None):
        if board in self.transition_t:
            node_data = self.transition_t[board]
        else:
            board.check_terminal_position()
            node_data = NodeData(board, self.evaluation_type())
            self.transition_t[board] = node_data

        node = anytree.Node(name, parent=parent, data=node_data)
        return node

    def update_root(self, board):
        self.root = self.create_node('root', copy.deepcopy(board))
        # self.transition_t.age(board.age)

    def take_action(self, action, node):
        new_board = copy.deepcopy(node.data.board)
        new_board.make_move(action)
        return self.create_node(action, new_board, parent=node)

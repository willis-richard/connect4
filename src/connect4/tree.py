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


class Connect4Tree():
    def __init__(self, evaluation_type, evaluate_fn, transition_t=None):
        self.evaluation_type = evaluation_type
        self.evaluate_position = evaluate_fn

        # transition_t is a map from board to NodeEvaluation
        self.transition_t = dict()
        # at least while aging is not a problem, a dict() is
        # faster than my other implementations
        # self.transition_t = TransitionTableDictOfDict() \
        #     if transition_t is None else transition_t

    def create_node(self, name, board, parent=None):
        if board in self.transition_t:
            node_data = NodeData(board, self.transition_t[board])
        else:
            evaluation = self.evaluation_type(board.check_terminal_position())
            node_data = NodeData(board, evaluation)
            self.transition_t[board] = evaluation

        node = anytree.Node(name, parent=parent, data=node_data)
        return node

    def update_root(self, board):
        self.root = self.create_node('root', copy.deepcopy(board))
        # self.transition_t.age(board.age)

    def take_action(self, action, node):
        new_board = copy.deepcopy(node.data.board)
        new_board.make_move(action)
        return self.create_node(action, new_board, parent=node)

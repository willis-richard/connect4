import anytree
import copy

from src.connect4.transition_table import TransitionTable


class NodeData():
    """Node data is a pair of a board and some evaluation data"""
    def __init__(self, board, node_eval=None):
        self.board = board
        self.node_eval = NodeEvaluation(board.check_terminal_position()) \
            if node_eval is None else node_eval

    def expandable(self):
        return self.node_eval.terminal_result is None


class NodeEvaluation():
    def __init__(self, terminal_result=None, evaluation=None, tree_value=None):
        self.terminal_result = terminal_result
        self.evaluation = evaluation
        self.tree_value = tree_value
        # self.visits = None
        # etc

    def get_value(self):
        if self.terminal_result is not None:
            return self.terminal_result
        elif self.tree_value is not None:
            return self.tree_value
        elif self.evaluation is not None:
            return self.evaluation
        raise RuntimeError("No NodeEvaluation value set")


class Connect4Tree():
    def __init__(self, evaluate_fn, transition_t=None):
        """transition_t is a map from board to NodeEvaluation"""
        self.evaluate_position = evaluate_fn
        self.transition_t = TransitionTable() \
            if transition_t is None else transition_t

    def create_node(self, name, board, parent=None):
        # FIXME: to confirm that my eq operator gives us a cache hit and and the 'is' operator isn't used instead
        if board in self.transition_t:
            node_data = NodeData(board, self.transition_t[board])
        else:
            node_data = NodeData(board)

        self.transition_t[board] = node_data.node_eval
        node = anytree.Node(name, parent=parent, data=node_data)
        return node

    def update_root(self, board):
        self.root = self.create_node('root', copy.deepcopy(board))
        self.transition_t.age(board.age)

    def take_action(self, action, node):
        new_board = copy.deepcopy(node.data.board)
        new_board.make_move(action)
        return self.create_node(action, new_board, parent=node)

    def expand_tree(self, node, plies):
        if plies == 0 or node.data.node_eval.terminal_result:
            return
        # create new nodes for all unexplored moves
        actions_not_taken = node.data.board.valid_moves().difference([c.name for c in node.children])
        for move in actions_not_taken:
            self.take_action(move, node)

        for child in node.children:
            self.expand_tree(child, plies - 1)

    def nega_max(self, node, plies, side):
        # https://en.wikipedia.org/wiki/Negamax
        if node.data.node_eval.terminal_result is not None:
            return node.data.node_eval.terminal_result
        if plies == 0:
            evaluation = self.evaluate_position(node.data.board)
            node.data.node_eval = NodeEvaluation(None, evaluation, None)
            return evaluation

        if side == 1:
            value = -2
            for child in node.children:
                value = max(value, self.nega_max(child, plies - 1, -side))
        else:
            value = 2
            for child in node.children:
                value = min(value, self.nega_max(child, plies - 1, -side))

        node.data.node_eval.tree_value = value
        return value

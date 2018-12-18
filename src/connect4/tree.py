import anytree
import copy


class NodeData():
    """Node data is a pair of a board and some evaluation data"""
    def __init__(self, board, node_eval=None):
        self.board = board
        self.node_eval = NodeEvaluation(board.check_terminal_position()) if node_eval is None else node_eval


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
    def __init__(self):
        """transition_t is a map from board to NodeEvaluation"""
        self.transition_t = dict()
        # self.root = self.create_node(copy.deepcopy(board))
        # can I clear out the tree each move by setting the new node as the root (and it's parent to None)
        # then delete the entire old tree?
        # if the positions live on in the transition table then is could be rosy

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
        # Don't be silly - just walk the two last moves if they exist.
        """new_root = anytree.search.find(self.root, lambda n: np.array_equal(n.name == board.move_history), maxlevel=2)
        if new_root:
            new_root.parent = None
        else:
            # delete entire tree?
            new_root = self.create_node(copy.deepcopy(board))

        self.root = new_root"""
        #self.prune_old_tree()
        #self.age_transition_tree()

    def expand_node(self, node, plies):
        if plies == 0 or node.data.node_eval.terminal_result:
            return
        # create new nodes for all unexplored moves
        actions_not_taken = node.data.board.valid_moves().difference([c.name for c in node.children])
        for move in actions_not_taken:
            new_board = copy.deepcopy(node.data.board)
            new_board.make_move(move)
            self.create_node(move, new_board, parent=node)

        for child in node.children:
            self.expand_node(child, plies - 1)

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

    def evaluate_position(self, board):
        return 0

    def prune_old_tree(self):
        return

    def age_transition_tree(self):
        return

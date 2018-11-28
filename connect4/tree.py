import anytree
import copy


class NodeData():
    """Node data is a pair of a board and some evaluation data"""
    def __init__(self, board, node_eval=None):
        self.board = board
        self.node_eval = NodeEvaluation() if node_eval is None else node_eval


class NodeEvaluation():
    def __init__(self, evaluation=None, tree_value=None):
        self.evaluation = evaluation
        self.tree_value = tree_value
        # self.visits = None
        # etc


class Connect4Tree():
    def __init__(self, board):
        """transition_t is a map from board to NodeEvaluation"""
        self.transition_t = dict()
        self.root = self.create_node(board)
        # can I clear out the tree each move by setting the new node as the root (and it's parent to None)
        # then delete the entire old tree?
        # if the positions live on in the transition table then is could be rosy

    def create_node(self, board, parent=None):
        # N.B. anytree.Node.name will be the move_history
        if board in self.transition_t:
            node_data = NodeData(board, self.transition_t[board])
        else:
            node_data = NodeData(board)

        self.transition_t[board] = node_data.node_eval
        node = anytree.Node(board.move_history, parent=parent, data=node_data)
        return node

    def update_root(self, board):
        new_root = anytree.search.find(self.root, lambda n: n.name == board.move_history, maxlevel=2)
        if new_root:
            print("Found root in tree")
            new_root.parent = None
        else:
            print("Did not find root in tree")
            new_root = self.create_node(board)
        self.root = new_root
        print("Set root to: ")
        new_root.data.board.display()
        #self.prune_old_tree()
        #self.age_transition_tree()

    def nega_max(self, node, half_moves):
        #node.data.board.check_terminal_position()
        #print("nega_max called with half moves = ", half_moves, "and board")
        #node.data.board.display()
        if half_moves > 0:
            # create new nodes for all unexplored moves
            actions_not_taken = node.data.board.valid_moves().difference([c.name[0] for c in node.children])
            for move in actions_not_taken:
                new_board = copy.deepcopy(node.data.board)
                new_board.make_move(move)
                self.create_node(new_board, parent=node)

            max_ = -2

            for child in node.children:
                print(type(child))
                score = -self.nega_max(child, half_moves - 1)
                if score > max_:
                    max_ = score

            node.data.node_eval.tree_value = max_
            return max_
        else:
            if node.data.node_eval.tree_value:
                return node.data.node_eval.tree_value
            evaluation = self.evaluate_position(node.data.board)
            node.data.node_eval.evaluation = evaluation
            #node.data.board.display()
            #print("evaluated board as: ", evaluation)
            return evaluation

    def explore_move(move):
        return

    def evaluate_position(self, board):
        value = board.check_terminal_position()
        return 0 if value is None else value

    def prune_old_tree(self):
        return

    def age_transition_tree(self):
        return

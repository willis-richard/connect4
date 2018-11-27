from connect4 import board

import anytree
import copy


class NodeData():
    """Node data is a pair of a board and some evaluation data"""
    def __init__(self, board, node_eval=None):
        self.board = board
        self.node_eval = node_eval

def NodeEvaluation():
    def __init__(self):
        self.evaluation = evaluation
        self.tree_value = None
        # self.visits = None
        # etc


class Connect4Tree():
    def __init__(self, board):
        """transition_t is a map from board to NodeEvaluation"""
        self.transition_t = dict()
        self.root = create_node(-1, board)
        # can I clear out the tree each move by setting the new node as the root (and it's parent to None)
        # then delete the entire old tree?
        # if the positions live on in the transition table then is could be rosy

    def create_node(self, move, board, parent=None):
        # N.B. anytree.Node.name will be the move_history
        if board in self.transition_t:
            node_data = NodeData(board, self.transition_t[board])
        else:
            node_data = NodeData(board)

        self.transition_t[board] = node_data
        node = anytree.Node(board.move_history, parent=parent, data=node_data)
        return node

    def update_root(self, board):
        new_root = anytree.search.find(self.root, lambda n: n.name == board.move_history, maxlevel=2)
        if new_root:
            new_root.parent = None
        else:
            new_root = self.create_node(board)
        self.root = new_root
        self.prune_old_tree()
        self.age_transitiion_tree()

    def expand_tree(self, node, half_moves):
        if half_moves:
            # first expand out the tree for all moves previously taken
            for node in anytree.iterator.LevelOrderIterator(node, maxlevel=1)[-1]:
                self.expand_tree(node, half_moves - 1)

            # then create new nodes for all unexplored moves
            actions_not_taken = node.data.board.valid_moves().difference([c[-1] for c in node.children])
            for move in actions_not_taken:
                new_board = copy.deepcopy(node.name)
                new_board.make_move(move)
                self.create_node(new_board, parent=node)

    def explore_move(move):
        return

    def evaluate_position(self, board):
        return board.check_terminal_position()

    def evaluate_tree(self):
        # apply min max
        return


    def prune_old_tree(self):
        return

    def age_transition_tree(self):
        return

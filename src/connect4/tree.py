import anytree
import copy

import numpy as np

from functools import partial


class NodeData():
    """Node data is a pair of a board and some evaluation data"""
    def __init__(self, board, evaluation):
        self.board = board
        self.valid_moves = board.valid_moves
        self.evaluation = evaluation

    def unexplored_moves(self, children):
        return self.valid_moves.difference([c.name for c in children])


class GridSearch():
    def __init__(self, depth):
        self.depth = depth

    @staticmethod
    def search(tree, board, side, depth):
        tree.expand_tree(tree.root, depth)
        tree.nega_max(tree.root, depth, side)

        moves = np.array([(n.name) for n in tree.root.children])
        values = np.array([(n.data.evaluation.get_value())
                           for n in tree.root.children])
        idx = np.argmax(values * side)
        best_move_value = values[idx]

        best_move = moves[idx]
        return best_move, best_move_value

    def get_search_fn(self):
        return partial(GridSearch.search,
                       depth=self.depth)

    class Evaluation():
        def __init__(self,
                     terminal_result=None,
                     evaluation=None,
                     tree_value=None):
            self.terminal_result = terminal_result
            self.evaluation = evaluation
            self.tree_value = tree_value

        def get_value(self):
            if self.terminal_result is not None:
                return self.terminal_result
            elif self.tree_value is not None:
                return self.tree_value
            elif self.evaluation is not None:
                return self.evaluation
            raise RuntimeError("No Evaluation value set")


class MCTS():
    @staticmethod
    def search(tree, board, side, simulations):
        def ucb_score(node, child):
            return 0

        def select_child(node):
            # guaranteed at least one non-terminally forcing child
            if not node.children:
                tree.expand_tree(node, 1)

            non_terminal_actions = set(node.data.valid_moves).\
                difference(node.data.terminal_children)
            non_terminal_children = [child for child in node.children if
                                     child.name in non_terminal_actions]
            _, child = max((ucb_score(node, child), child)
                           for child in non_terminal_children)

            return child

        for _ in range(simulations):
            node = tree.root
            while node.evaluated():
                node = select_child(node)

            if node.visited():
                tree.expand_tree(node, 1)
                # action = select_action(node)
                # node = tree.take_action(action, node)

            # while node.expandable():
            #     action = select_action(node)
            #     actions_explored = [c.name for c in node.children]
            #     if action in actions_explored:
            #         node = node.children[actions_explored == action]
            #     else:
            #         break

            # node = tree.take_action(action, node)

            node.evaluate()
            tree.backpropagage(node)

            return 0, 0

    class Evaluation():
        def __init__(self, board):
            self.Q = board

        def evaluated(self):
            return self.evaluation.terminal_result is None


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

    def expand_tree(self, node, plies):
        if plies == 0 or node.data.evaluation.terminal_result:
            return
        # create new nodes for all unexplored moves
        for move in node.data.unexplored_moves(node.children):
            self.take_action(move, node)

        for child in node.children:
            self.expand_tree(child, plies - 1)

    def nega_max(self, node, plies, side):
        # https://en.wikipedia.org/wiki/Negamax
        if node.data.evaluation.terminal_result is not None:
            return node.data.evaluation.terminal_result
        if plies == 0:
            evaluation = self.evaluate_position(node.data.board)
            node.data.evaluation = self.evaluation_type(None,
                                                        evaluation,
                                                        None)
            return evaluation

        if side == 1:
            value = -2
            for child in node.children:
                value = max(value, self.nega_max(child, plies - 1, -side))
        else:
            value = 2
            for child in node.children:
                value = min(value, self.nega_max(child, plies - 1, -side))

        node.data.evaluation.tree_value = value
        return value

from src.connect4.utils import Connect4Stats as info

import math
import numpy as np

from functools import partial


class GridSearch():
    def __init__(self, depth):
        self.depth = depth

    def get_search_fn(self):
        return partial(grid_search,
                       depth=self.depth)

    def get_evaluate_position_fn(self):
        return evaluate_position_centre

    class Evaluation():
        def __init__(self):
            self.tree_value = None
            self.position_evaluation = None

        @property
        def value(self):
            if self.tree_value is not None:
                return self.tree_value
            elif self.position_evaluation is not None:
                return self.position_evaluation
            raise RuntimeError("No Evaluation value set")

        def __repr__(self):
            return "tree_value: " + str(self.tree_value) + \
                ",  position_evaluation: " + str(self.position_evaluation)


class MCTS():
    def __init__(self, simulations):
        self.simulations = simulations

    def get_search_fn(self):
        return partial(mcts_search,
                       simulations=self.simulations)

    def get_evaluate_position_fn(self):
        return evaluate_position_centre

    class Evaluation():
        def __init__(self):
            self._prior = None
            self.to_play = -1
            self.value_sum = 0
            self.visit_count = 0

        def evaluated(self):
            return self.visit_count != 0

        @property
        def prior(self):
            assert self._prior is not None
            return self._prior

        @prior.setter
        def prior(self, prior):
            assert self._prior is None
            self._prior = prior

        @property
        def value(self):
            return self.prior


def evaluate_position_centre(board):
    # return np.sum(np.multiply(board.o_pieces, info.value_grid) -
    # np.multiply(board.x_pieces, info.value_grid)) \
    # / float(info.value_grid_sum)
    return (np.einsum('ij,ij', board.o_pieces, info.value_grid)
            - np.einsum('ij,ij', board.x_pieces, info.value_grid)) \
            / float(info.value_grid_sum)


def grid_search(tree, board, side, depth):
    expand_tree(tree, tree.root, depth)
    nega_max(tree.root, depth, side)

    moves = np.array([(n.name) for n in tree.root.children])
    values = np.array([(n.data.value)
                       for n in tree.root.children])
    idx = np.argmax(values * side)
    best_move_value = values[idx]

    best_move = moves[idx]
    return best_move, best_move_value


def expand_tree(tree, node, plies):
    if plies == 0 or node.data.board.result:
        return
    # create new nodes for all unexplored moves
    for move in node.data.unexplored_moves(node.children):
        tree.take_action(move, node)

    for child in node.children:
        expand_tree(tree, child, plies - 1)


def nega_max(node, plies, side):
    # https://en.wikipedia.org/wiki/Negamax
    if node.data.board.result is not None:
        return node.data.board.result
    if plies == 0:
        position_evaluation = evaluate_position_centre(node.data.board)
        node.data.evaluation.position_evaluation = position_evaluation
        return position_evaluation

    if side == 1:
        value = -2
        for child in node.children:
            value = max(value, nega_max(child, plies - 1, -side))
    else:
        value = 2
        for child in node.children:
            value = min(value, nega_max(child, plies - 1, -side))

    node.data.evaluation.tree_value = value
    return value


def mcts_search(tree, board, side, simulations):
    for _ in range(simulations):
        # it is possible the root node has not been evaluated
        if not tree.root.evaluated():
            evaluate_nn(tree.root)

        node = tree.root

        while node.children:
            node = select_child(tree, node)

        evaluate_nn(node)

        tree.backpropagage(node)

        set_child_priors(node)

        # node = tree.take_action(action, node)

        # while node.expandable():
        #     action = select_action(node)
        #     actions_explored = [c.name for c in node.children]
        #     if action in actions_explored:
        #         node = node.children[actions_explored == action]
        #     else:
        #         break

        # node = tree.take_action(action, node)
        return select_action(tree)


def evaluate_nn(node):
    return 0


def ucb_score(node, child):
    return 0


def select_child(tree, node):
    # guaranteed at least one non-terminally forcing child
    if not node.children:
        expand_tree(tree, node, 1)

    non_terminal_actions = set(node.data.valid_moves).\
        difference(node.data.terminal_children)
    non_terminal_children = [child for child in node.children if
                             child.name in non_terminal_actions]
    _, child = max((MCTS.ucb_score(node, child), child)
                   for child in non_terminal_children)

    return child


def set_child_priors(tree, node):
    expand_tree(tree, node, 1)
    policy = {a: math.exp(node.data.evaluation.policy_logits[a.name])
              for a in node.data.valid_moves}
    policy_sum = sum(policy.itervalues())
    for action, p in policy.iteritems():
        node.children[action].data.evaluation.prior = p / policy_sum


def policy_logits(a):
    return 0


def select_action(tree):
    return 0

from src.connect4.board import Board
from src.connect4.tree import Tree

from src.connect4.utils import Connect4Stats as info

from anytree import Node
from functools import partial

import math

import numpy as np


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
        self.config = MCTS.Config(simulations)

    class Config():
        def __init__(self, simulations):
            self.simulations = simulations
            self.pb_c_base = 0
            self.pb_c_init = 0

    def get_search_fn(self):
        return partial(mcts_search,
                       config=self.config)

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
            if self.visit_count:
                return self.value_sum / self.visit_count
            # return self.prior
            # FIXME: confirm this behaviour
            return 0


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

    if not node.children:
        for move in node.data.valid_moves:
            tree.create_child(move, node)

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


def mcts_search(config: MCTS.Config, tree: Tree, board: Board, side):
    for _ in range(config.simulations):
        node = tree.root

        while node.children:
            node = select_child(config, tree, node)

        evaluate_nn(config, node)
        set_child_priors(config, node)

        tree.backpropagage(config, node)

        return select_action(config, tree)


def evaluate_nn(config: MCTS.Config, node: Node):
    # FIXME: implement. N.B. google one is just some the expanding of the
    # children

    return 0


def ucb_score(config: MCTS.Config, node: Node, child: Node):
    pb_c = math.log((node.visit_count + config.pb_c_base + 1) /
                    config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(node.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value
    return prior_score + value_score


def select_child(config: MCTS.Config, tree: Tree, node: Node):
    non_terminal_children = [child for child in node.children if
                             child.name in node.data.non_terminal_actions]

    _, child = max((MCTS.ucb_score(node, child), child)
                   for child in non_terminal_children)

    return child


def set_child_priors(config: MCTS.Config, tree: Tree, node: Node):
    expand_tree(tree, node, 1)
    policy = {a: math.exp(node.data.evaluation.policy_logits[a.name])
              for a in node.data.valid_moves}
    policy_sum = sum(policy.itervalues())
    for action, p in policy.iteritems():
        node.children[action].data.evaluation.prior = p / policy_sum


def policy_logits(config: MCTS.Config, a):
    return 0


def select_action(config: MCTS.Config, tree: Tree):
    # FIXME: removed softmax thing (would check game move history)
    return max([(child.visit_count, action)
                for action, child in tree.root.children])


def backpropagate(node: Node, value: float, to_play):
    while node.is_leaf():
        node = node.parent
        # FIXME: to confirm
        node.value_sum += value if node.data.to_play == to_play else (1 - value)
        node.visit_count += 1

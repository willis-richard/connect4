from src.connect4.board import Board
from src.connect4.tree import Tree

from src.connect4.utils import Connect4Stats as info
from src.connect4.utils import Side

from anytree import Node
from functools import partial
from typing import Callable

import math

import numpy as np


class GridSearch():
    def __init__(self, plies):
        self.plies = plies

    def get_search_fn(self):
        return partial(grid_search,
                       plies=self.plies,
                       evaluate_fn=evaluate_position_centre)

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
                       config=self.config,
                       evaluate_fn=partial(evaluate_nn(config=self.config)))

    class Evaluation():
        def __init__(self):
            self._prior = None
            self.to_play = -1
            self.value_sum = 0
            self.visit_count = 0
            self.policy_logits = None

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

        def __repr__(self):
            return "prior: " + str(self.prior) + \
                ",  to_play: " + str(self.to_play) + \
                ",  value_sum: " + str(self.value_sum) + \
                ",  visit_count: " + str(self.visit_count)


def evaluate_position_centre(node: Node):
    # return np.sum(np.multiply(board.o_pieces, info.value_grid) -
    # np.multiply(board.x_pieces, info.value_grid)) \
    # / float(info.value_grid_sum)
    board = node.data.board
    node.data.evaluation.position_evaluation = \
        (np.einsum('ij,ij', board.o_pieces, info.value_grid)
         - np.einsum('ij,ij', board.x_pieces, info.value_grid)) \
        / float(info.value_grid_sum)


def grid_search(tree: Tree,
                board: Board,
                side: Side,
                plies,
                evaluate_fn: Callable[[Node], None]):
    expand_tree(tree, tree.root, plies)
    nega_max(tree.root, plies, side, evaluate_fn)

    moves = np.array([(n.name) for n in tree.root.children])
    values = np.array([(n.data.value)
                       for n in tree.root.children])
    idx = np.argmax(values * side)
    best_move_value = values[idx]

    best_move = moves[idx]
    return best_move, best_move_value


def expand_tree(tree: Tree,
                node: Node,
                plies):
    if plies == 0 or node.data.board.result:
        return

    if not node.children:
        for move in node.data.valid_moves:
            tree.create_child(move, node)

    for child in node.children:
        expand_tree(tree, child, plies - 1)


def nega_max(node: Node,
             plies,
             side: Side,
             evaluate_fn: Callable[[Node], None]):
    # https://en.wikipedia.org/wiki/Negamax
    if node.data.board.result is not None:
        return node.data.board.result
    if plies == 0:
        evaluate_fn(node)
        return node.data.evaluation.position_evaluation

    if side == side.o:
        value = -2
        for child in node.children:
            value = max(value, nega_max(child, plies - 1, -side, evaluate_fn))
    else:
        value = 2
        for child in node.children:
            value = min(value, nega_max(child, plies - 1, -side, evaluate_fn))

    node.data.evaluation.tree_value = value
    return value


def mcts_search(config: MCTS.Config,
                evaluate_fn: Callable[[Node], None],
                tree: Tree,
                board: Board,
                side: Side):
    for _ in range(config.simulations):
        node = tree.root

        while node.children:
            node = select_child(config, tree, node)

        value, priors = evaluate_nn(config, node)
        set_child_priors(config, node)

        tree.backpropagage(config, node)

        return select_action(config, tree)


def evaluate_nn(config: MCTS.Config, node: Node):
    # FIXME: implement. N.B. google one is just some the expanding of the
    # children
    node.data.evaluation.value_sum += evaluate_position_centre(node.data.board)
    node.data.evaluation.visit_count += 1
    node.data.evaluation.policy_logits = {a: 0 for a in range(info.width)}


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


def set_child_priors(config: MCTS.Config,
                     tree: Tree,
                     node: Node):
    expand_tree(tree, node, 1)
    policy = {a: math.exp(node.data.evaluation.policy_logits[a.name])
              for a in node.data.valid_moves}
    policy_sum = sum(policy.itervalues())
    for action, p in policy.iteritems():
        node.children[action].data.evaluation.prior = p / policy_sum


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

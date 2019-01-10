from src.connect4.board import Board
from src.connect4.tree import Tree

from src.connect4.utils import Connect4Stats as info
from src.connect4.utils import Side

from anytree import Node
from functools import partial
from typing import Callable, Dict, Tuple

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
    class Config():
        def __init__(self, simulations, cpuct=None):
            self.simulations = simulations
            self.pb_c_base = 0
            self.pb_c_init = 0
            self.cpuct = 2.4 if cpuct is None else cpuct

    def __init__(self, config: Config):
        self.config = config

    def get_search_fn(self):
        return partial(mcts_search,
                       config=self.config,
                       evaluate_fn=partial(evaluate_nn, config=self.config))

    class Evaluation():
        def __init__(self):
            # FIXME: how do transpositions impact prior? Different parents will
            # give different priors...
            self._prior = None
            self.value_sum = 0.0
            self.visit_count = 0
            self.policy_logits = None

        def evaluated(self):
            return self.visit_count != 0

        def add_value(self, value: float):
            self.value_sum += value
            self.visit_count += 1

        @property
        def prior(self) -> float:
            assert self._prior is not None
            return self._prior

        @prior.setter
        def prior(self, prior: float):
            assert self._prior is None
            self._prior = prior

        @property
        def value(self) -> float:
            assert self.evaluated()
            # FIXME: responsibility of the node to know who's turn it is
            return self.value_sum / self.visit_count

        def __repr__(self):
            return "prior: " + str(self._prior) + \
                ",  value_sum: " + str(self.value_sum) + \
                ",  visit_count: " + str(self.visit_count) + \
                ", policy_logits: " + str(self.policy_logits)


def evaluate_position_centre(node: Node):
    # return np.sum(np.multiply(board.o_pieces, info.value_grid) -
    # np.multiply(board.x_pieces, info.value_grid)) \
    # / float(info.value_grid_sum)
    board = node.data.board
    return \
        (np.einsum('ij,ij', board.o_pieces, info.value_grid)
         - np.einsum('ij,ij', board.x_pieces, info.value_grid)) \
        / float(info.value_grid_sum)


def grid_search(tree: Tree,
                board: Board,
                side: Side,
                plies,
                evaluate_fn: Callable[[Node], float]):
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
             evaluate_fn: Callable[[Node], float]):
    # https://en.wikipedia.org/wiki/Negamax
    if node.data.board.result is not None:
        return node.data.board.result
    if plies == 0:
        value = evaluate_fn(node)
        node.data.evaluation.position_evaluation = value
        return value

    if side == Side.o:
        value = -2
        for child in node.children:
            value = max(value,
                        nega_max(child, plies - 1, Side(-side), evaluate_fn))
    else:
        value = 2
        for child in node.children:
            value = min(value,
                        nega_max(child, plies - 1, Side(-side), evaluate_fn))

    node.data.evaluation.tree_value = value
    return value


def mcts_search(config: MCTS.Config,
                evaluate_fn: Callable[[Node], Tuple[float, Dict[int, float]]],
                tree: Tree,
                board: Board,
                side: Side):
    for _ in range(config.simulations):
        node = tree.root

        # FIXME: node.data.evaluated() ?
        while node.children:
            node = select_child(config, tree, node)

        value, policy_logits = evaluate_nn(config, node)
        node.data.evaluation.add_value(value)
        node.data.evaluation.policy_logits = policy_logits
        set_child_priors(tree, node)

        backpropagate(node, value, side)

    return select_action(config, tree)


def evaluate_nn(config: MCTS.Config, node: Node):
    # FIXME: remove when sure positions aren't evaluated multiple times
    assert not node.data.evaluated()
    # FIXME: implement. N.B. google one is just some the expanding of the
    # children
    value = evaluate_position_centre(node)
    print ("posn value: ", value)
    policy_logits = {a: 1 for a in range(info.width)}
    return value, policy_logits


def ucb_score(config: MCTS.Config, node: Node, child: Node):
    # pb_c = math.log((node.visit_count + config.pb_c_base + 1) /
    #                 config.pb_c_base) + config.pb_c_init
    # pb_c *= math.sqrt(node.visit_count) / (child.visit_count + 1)
    pb_c = config.cpuct

    prior_score = pb_c * child.data.evaluation.prior
    value_score = child.data.evaluation.value
    print("child, prior, value, total:  ", child.name, prior_score, value_score, prior_score + value_score)
    return prior_score + value_score


def select_child(config: MCTS.Config, tree: Tree, node: Node):
    non_terminal_children = [child for child in node.children if
                             child.name in node.data.non_terminal_moves]
    _, child_name = max((ucb_score(config, node, child), child.name)
                        for child in non_terminal_children)
    child = node.children[[c.name for c in node.children].index(child_name)]

    return child


def set_child_priors(tree: Tree, node: Node):
    expand_tree(tree, node, 1)
    policy = {a: math.exp(node.data.evaluation.policy_logits[a])
              for a in node.data.valid_moves}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action].data.evaluation.prior = p / policy_sum


def select_action(config: MCTS.Config, tree: Tree):
    # FIXME: removed softmax thing (would check game move history)
    _, action = max(((c.data.evaluation.visit_count, c.name)
                    for c in tree.root.children))

    # FIXME: what is the value of that child? avg value_sum?
    return action, 0


def backpropagate(node: Node,
                  value: float,
                  side: Side):
    while not node.is_root:
        node = node.parent
        # FIXME: to confirm
        node.data.evaluation.add_value(value)
            # value if node.data.board._player_to_move == side else (1 - value))

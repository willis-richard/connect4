from oinkoink.board import Board
from oinkoink.evaluators import Evaluator
from oinkoink.player import BasePlayer
from oinkoink.tree import Tree
from oinkoink.utils import Connect4Stats as info

from anytree import Node
import math
import numpy as np
from typing import Callable, List, Set, Tuple


class MCTSConfig():
    def __init__(self,
                 simulations: int,
                 pb_c_base: int = 19652,
                 pb_c_init: float = 1.25,
                 root_dirichlet_alpha: float = 0.0,
                 root_exploration_fraction: float = 0.0,
                 num_sampling_moves=0):
        self.simulations = simulations
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction
        self.num_sampling_moves = num_sampling_moves


class PositionEvaluation():
    def __init__(self,
                 value: float,
                 prior: np.ndarray):
        self.value = value
        self.prior = prior

    def __float__(self):
        return float(self.value)

    def __str__(self):
        return str("{:.4f}".format(self.__float__()))

    def __repr__(self):
        return "position_value: " + str(self.value) + \
            ", prior: " + str(self.prior)


class SearchEvaluation():
    def __init__(self):
        self.value_sum = 0.0
        self.visit_count = 0

    def add(self, value: float):
        self.value_sum += value
        self.visit_count += 1

    def __float__(self):
        assert self.visit_count != 0
        return float(self.value_sum / self.visit_count)

    def __str__(self):
        return str("{:.4f}".format(self.__float__()))

    def __repr__(self):
        return "value: " + str(self.__float__()) + \
            ",  value_sum: " + str(self.value_sum) + \
            ",  visit_count: " + str(self.visit_count)


class MCTS(BasePlayer):
    def __init__(self,
                 name: str,
                 config: MCTSConfig,
                 evaluator: Evaluator):
        super().__init__(name)
        self.config = config
        self.evaluator = evaluator

    def make_move(self, board):
        tree = search(self.config, board, self.evaluator)

        if board.age < self.config.num_sampling_moves:
            child = tree.sample_value_fn(lambda x: x ** 2)
        else:
            child = tree.best_move()

        board.make_move(child.name)

        return child.name, child.data.absolute_value, tree

    def __str__(self):
        return super().__str__() + ", type: Computer"


def search(config: MCTSConfig,
           board: Board,
           evaluator: Callable[[Board],
                               Tuple[float, List[float]]]):
    tree = Tree(board)

    # First evaluate root and add noise
    evaluate_node(tree, tree.root, evaluator)
    tree.root.data.position_value.prior = add_exploration_noise(
        config,
        tree.root.data.position_value.prior,
        tree.root.data.valid_moves)

    for _ in range(config.simulations):
        node = tree.root

        while node.children:
            node = select_child(config, tree, node)

        # previously evaluated, so expand
        if node.data.position_value is not None:
            tree.expand_node(node, 1)
            node = select_child(config, tree, node)

        value = evaluate_node(tree, node, evaluator)

        backpropagate(node, value)
    return tree


def evaluate_node(tree: Tree, node: Node, evaluator):
    if node.data.board.result is not None:
        value = node.data.board.result.value
        if node.data.search_value is None:
            node.data.search_value = SearchEvaluation()
    else:
        value, prior = evaluator(node.data.board)
        normalise(node.data.valid_moves, prior)
        node.data.position_value = PositionEvaluation(value, prior)
        node.data.search_value = SearchEvaluation()
    node.data.search_value.add(value)
    return value


def select_child(config: MCTSConfig,
                 tree: Tree,
                 node: Node):
    _, child = max((ucb_score(config, node, c), c)
                   for c in node.children)

    return child


def ucb_score(config: MCTSConfig,
              node: Node,
              child: Node):
    pb_c = math.log(
        (node.data.search_value.visit_count + config.pb_c_base + 1) /
        config.pb_c_base) + config.pb_c_init
    child_visit_count = 0 if child.data.search_value is None \
                        else child.data.search_value.visit_count
    pb_c = pb_c * (
        math.sqrt(node.data.search_value.visit_count)
        / (child_visit_count + 1))

    prior_score = pb_c * node.data.position_value.prior[child.name]
    value_score = child.data.value(node.data.board.player_to_move)
    return prior_score + value_score


def backpropagate(node: Node,
                  value: float):
    while not node.is_root:
        node = node.parent
        node.data.search_value.add(value)


def add_exploration_noise(config: MCTSConfig,
                          prior: np.ndarray,
                          valid_moves: Set):
    if config.root_dirichlet_alpha and config.root_exploration_fraction:
        noise = np.random.gamma(config.root_dirichlet_alpha,
                                1,
                                info.width)
        normalise(valid_moves, noise)
        frac = config.root_exploration_fraction
        prior = prior * (1 - frac) + noise * frac
    return prior


def add_exploration_noise_non_normalised(config: MCTSConfig,
                                         prior: np.ndarray,
                                         valid_moves: Set):
    if config.root_dirichlet_alpha and config.root_exploration_fraction:
        noise = np.random.gamma(config.root_dirichlet_alpha,
                                1,
                                len(valid_moves))
        for a, n in zip(valid_moves, noise):
            frac = config.root_exploration_fraction
            prior[a] = prior[a] * (1 - frac) + n * frac
    return prior


def normalise(valid_moves: Set, prior: np.ndarray):
    """This function works in-place, but must have a float-array prior"""
    invalid_moves = set(range(info.width)).difference(valid_moves)
    if invalid_moves:
        np.put(prior, list(invalid_moves), 0.0)
    prior /= np.sum(prior)

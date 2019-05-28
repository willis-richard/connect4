from connect4.board_c import Board
from connect4.evaluators import Evaluator
from connect4.player import BasePlayer
from connect4.tree import BaseNodeData, Tree
from connect4.utils import Connect4Stats as info
from connect4.utils import (same_side,
                            Side,
                            value_to_side,
                            Result)

from anytree import Node
from collections import namedtuple
import math
import numpy as np
from typing import Callable, List, Optional, Set, Tuple


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
        return str(self.__float__())

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
        return str(self.__float__())

    def __repr__(self):
        return "value: " + str(self.__float__()) + \
            ",  value_sum: " + str(self.value_sum) + \
            ",  visit_count: " + str(self.visit_count)


TerminalResult = namedtuple('TerminalResult', ['result', 'age'])


class NodeData(BaseNodeData):
    def __init__(self, board: Board):
        super().__init__(board)
        self.terminal_result: Optional[TerminalResult] = None
        self.terminal_moves: Set[int] = set()
        self.search_value = SearchEvaluation()

    @property
    def search_value(self):
        if self._search_value.visit_count == 0:
            return None
        else:
            return self._search_value

    @search_value.setter
    def search_value(self, value):
        if isinstance(value, SearchEvaluation):
            self._search_value = value

    @property
    def non_terminal_moves(self):
        return self.valid_moves.difference(self.terminal_moves)

    def value(self, side: Side):
        if self.terminal_result is not None:
            return value_to_side(float(self.terminal_result.result.value),
                                 side)
        return super().value(side)

    def __str__(self):
        return "terminal_result: " + str(self.terminal_result) + \
            "  " + super().__str__()

    def __repr__(self):
        return super().__repr__() + \
            "   terminal_result: " + str(self.terminal_result) + \
            ",  terminal_moves: " + str(self.terminal_moves)


class MCTS(BasePlayer):
    def __init__(self,
                 name: str,
                 config: MCTSConfig,
                 evaluator: Evaluator):
        super().__init__(name)
        self.config = config
        self.evaluator = evaluator

    def make_move(self, board):
        tree = Tree(board,
                    NodeData)

        search(self.config, tree, self.evaluator)

        if board.age >= self.config.num_sampling_moves:
            move, value = self.select_best_move(tree)
        else:
            move, value = tree.select_softmax_move()

        board.make_move(move)
        # Note that because we select the action greedily, the value of the root is equal to the perceived value of the 'best value' child
        # We need to return the value of the position to 'player_o'
        return move, value, tree

    def select_best_move(self, tree) -> Tuple[int, float]:
        if tree.root.data.terminal_result is None:
            move, value = tree.select_best_move()
        # choose shortest win
        elif same_side(tree.root.data.terminal_result.result, tree.side):
            value, _, move = max((tree.get_node_value(c),
                                  info.area - c.data.terminal_result.age,
                                  c.name)
                                 for c in tree.root.children
                                 if c.name in tree.root.data.terminal_moves)
        # else longest loss (or draw = 42)
        else:
            value, _, move = max((tree.get_node_value(c),
                                  c.data.terminal_result.age,
                                  c.name)
                                 for c in tree.root.children
                                 if c.name in tree.root.data.terminal_moves)

        absolute_value = value_to_side(value, tree.side)

        return move, absolute_value

    def __str__(self):
        return super().__str__() + ", type: Computer"


def search(config: MCTSConfig,
           tree: Tree,
           evaluator: Callable[[Board],
                               Tuple[float, List[float]]]):
    # First evaluate root and add noise
    root_data = tree.root.data
    value, root_prior = evaluator(root_data.board)
    # later we will return the root prior to it's true value so the transition table is accurate
    noisy_prior = add_exploration_noise(config,
                                        root_prior,
                                        root_data.valid_moves)
    tree.root.data.position_value = PositionEvaluation(value,
                                                       noisy_prior)

    for _ in range(config.simulations):
        if not tree.root.data.non_terminal_moves:
            break

        node = tree.root

        while node.children:
            node = select_child(config, tree, node)

        # previously evaluated, so expand
        if node.data.position_value is not None:
            tree.expand_node(node, 1)
            node = select_child(config, tree, node)

        board = node.data.board
        # encountered a new terminal position
        if board.result is not None:
            node = backpropagate_terminal(node,
                                          board.result,
                                          board.age)
            value = board.result.value
        else:
            value, prior = evaluator(board)
            node.data.position_value = PositionEvaluation(value, prior)

        backpropagate(node, value)

    # Return the root prior to it's true value
    tree.root.data.position_value.prior = root_prior
    return


def select_child(config: MCTSConfig,
                 tree: Tree,
                 node: Node):
    non_terminal_children = [child for child in node.children if
                             child.name not in node.data.terminal_moves]
    assert non_terminal_children

    if len(non_terminal_children) == 1:
        return non_terminal_children[0]

    _, idx = max((ucb_score(config, node, child), i)
                 for i, child in enumerate(non_terminal_children))

    return non_terminal_children[idx]


def ucb_score(config: MCTSConfig,
              node: Node,
              child: Node):
    pb_c = math.log((node.data._search_value.visit_count + config.pb_c_base + 1) /
                    config.pb_c_base) + config.pb_c_init
    pb_c = pb_c * (
        math.sqrt(node.data._search_value.visit_count)
        / (child.data._search_value.visit_count + 1))

    prior_score = pb_c * node.data.position_value.prior[child.name]
    value_score = child.data.value(node.data.board.player_to_move)
    return prior_score + value_score


def backpropagate_terminal(node: Node,
                           result: Result,
                           age: int):
    node.data.terminal_result = TerminalResult(result, age)

    if node.is_root:
        return node

    node.parent.data.terminal_moves.add(node.name)
    node = node.parent

    # The side to move can choose a win - they choose the quickest forcing line
    if same_side(result, node.data.board.player_to_move):
        age = min((c.data.terminal_result.age
                   for c in node.children
                   if c.name in node.data.terminal_moves))
        return backpropagate_terminal(node, result, age)
    # The parent only has continuations that lead to terminal results
    # None of them can be winning for this node, because otherwise we would
    # have previously set the terminal value of this node, and never visited it
    # again in the mcts
    elif not node.data.non_terminal_moves:
        # the parent will now choose a draw if possible
        # if forced to choose a loss, will choose the longest one
        # N.B. in connect4 all draws have age 42
        _, age, idx = max((value_to_side(c.data.terminal_result.result.value,
                                         node.data.board.player_to_move),
                           c.data.terminal_result.age,
                           i)
                          for i, c in enumerate(node.children))
        result = node.children[idx].data.terminal_result.result
        return backpropagate_terminal(node, result, age)

    # nothing to do
    return node


def backpropagate(node: Node,
                  value: float):
    node.data._search_value.add(value)
    while not node.is_root:
        node = node.parent
        node.data._search_value.add(value)


def add_exploration_noise(config: MCTSConfig,
                          prior: np.ndarray,
                          valid_moves: Set):
    if config.root_dirichlet_alpha and config.root_exploration_fraction:
        noise = np.random.gamma(config.root_dirichlet_alpha,
                                1,
                                info.width)
        frac = config.root_exploration_fraction
        prior = prior * (1 - frac) + noise * frac
        prior = normalise_prior(valid_moves, prior)
    return prior


def normalise_prior(valid_moves: Set, prior: np.ndarray):
    invalid_moves = set(range(info.width)).difference(valid_moves)
    if invalid_moves:
        np.put(prior, list(invalid_moves), 0.0)
    prior = prior / np.sum(prior)
    return prior

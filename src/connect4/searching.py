from src.connect4.board import Board
from src.connect4.tree import BaseNodeData, Tree
from src.connect4.utils import Connect4Stats as info
from src.connect4.utils import (same_side,
                                Side,
                                Result,
                                value_to_side)

from src.connect4.neural.network import Model

from anytree import Node
from functools import partial
from scipy.special import softmax
from typing import Callable, Dict, Optional, Tuple

import math

import numpy as np


class MCTS():
    class Config():
        def __init__(self, simulations, cpuct=None):
            self.simulations = simulations
            self.pb_c_base = 0
            self.pb_c_init = 0
            self.cpuct = 3.4 if cpuct is None else cpuct

    def __init__(self, config: Config):
        self.config = config

    def get_search_fn(self, model=None):
        if model is None:
            fn = evaluate_centre_with_prior
        else:
            fn = partial(evaluate_nn, model=model)

        return partial(mcts_search,
                       config=self.config,
                       evaluate_fn=fn)

    class PositionEvaluation():
        def __init__(self):
            self.value = None
            self.policy_logits = None

        def update_value(self,
                         value: Tuple[float, Dict]):
            self.value, self.policy_logits = value

        def __repr__(self):
            return "position_value: " + str(self.value) + \
                ", policy_logits: " + str(self.policy_logits)

    class SearchEvaluation():
        def __init__(self):
            # Although this is known at construction, I don't think the MCTS
            # wants to know this ahead of time
            self.terminal_result = None
            self.value_sum = 0.0
            self.visit_count = 0

        def update_value(self, value: float):
            self.value_sum += value
            self.visit_count += 1

        @property
        def value(self):
            if self.terminal_result is not None:
                return self.terminal_result[0].value
            if self.visit_count == 0:
                return 0
            return self.value_sum / self.visit_count

        def __repr__(self):
            return "terminal_result: " + str(self.terminal_result) + \
                ",  value_sum: " + str(self.value_sum) + \
                ",  visit_count: " + str(self.visit_count)

    class NodeData(BaseNodeData):
        def __init__(self,
                     board: Board,
                     position_evaluation):#: MCTS.PositionEvaluation):
            super().__init__(board,
                             position_evaluation,
                             MCTS.SearchEvaluation())
            self.non_terminal_moves = self.valid_moves.copy()
            # Necessary to stop mcts 'cheating' and knowing the value of nodes from
            # the transition table before it has selected them
            self.evaluated = False

        def update_terminal_result(self,
                                   result: Result,
                                   age: int):
            self.search_evaluation.terminal_result = (result, age)

        def add_terminal_move(self, move):
            self.non_terminal_moves.remove(move)

        @property
        def value(self):
            if not self.evaluated:
                # position is unknown - assume lost
                return 0
            return super().value

        def __repr__(self):
            return super().__repr__() + \
                ",  non_terminal_moves: " + str(self.non_terminal_moves) + \
                ",  evaluated: ", str(self.evaluated)


def evaluate_centre(node: Node):
    board = node.data.board
    value = 0.5 + \
        (np.einsum('ij,ij', board.o_pieces, info.value_grid)
         - np.einsum('ij,ij', board.x_pieces, info.value_grid)) \
        / float(info.value_grid_sum)
    return value


def evaluate_centre_with_prior(node: Node):
    value = evaluate_centre(node)
    return value, info.policy_logits



def mcts_search(config: MCTS.Config,
                evaluate_fn: Callable[[Node],
                                      Tuple[float, Dict[int, float]]],
                tree: Tree,
                board: Board):
    for _ in range(config.simulations):
        node = tree.root

        if not tree.root.data.non_terminal_moves:
            break

        # FIXME: node.data.evaluated() ?
        while node.children:
            node = select_child(config, tree, node)

        if node.data.evaluated:
            tree.expand_node(node, 1)
            node = select_child(config, tree, node)

        node.data.evaluated = True

        if node.data.board.result is not None:
            value = node.data.board.result.value
            node = backpropagate_terminal(node,
                                          node.data.board.result,
                                          node.data.board.age)
        else:
            # position may be in the transpositions table
            if node.data.position_evaluation.value is None:
                value, prior = evaluate_fn(node)
                prior = normalise_prior(node.data.valid_moves,
                                        prior)
                node.data.update_position_value((value, prior))
            value = node.data.position_evaluation.value
            # tree.expand_node(node, 1)

        backpropagate(node, value)

    return


def evaluate_nn(node: Node,
                model: Model):
    value, prior = model(node.data.board)
    value = value.cpu()
    value = value.view(-1)
    value = value.data.numpy()
    # prior = prior.cpu()
    # prior = prior.view(-1)
    # prior = prior.data.numpy()
    # prior = softmax(prior)
    prior = info.policy_logits
    return value, prior


def ucb_score(config: MCTS.Config,
              tree: Tree,
              node: Node,
              child: Node):
    # pb_c = math.log((node.visit_count + config.pb_c_base + 1) /
    #                 config.pb_c_base) + config.pb_c_init
    # pb_c *= math.sqrt(node.visit_count) / (child.visit_count + 1)
    # pb_c = config.cpuct
    pb_c = config.cpuct * math.sqrt(node.data.search_evaluation.visit_count) \
           / (child.data.search_evaluation.visit_count + 1)

    prior_score = pb_c * node.data.position_evaluation.policy_logits[child.name]
    value_score = tree.get_node_value(child)
    return prior_score + value_score


def normalise_prior(valid_moves, policy_logits):
    invalid_moves = set(range(info.width)).difference(valid_moves)
    if invalid_moves:
        for a in invalid_moves:
            policy_logits[a] = 0.0
    policy_logits = policy_logits / np.sum(policy_logits)
    return policy_logits


def select_child(config: MCTS.Config,
                 tree: Tree,
                 node: Node):
    non_terminal_children = [child for child in node.children if
                             child.name in node.data.non_terminal_moves]
    _, child = max((ucb_score(config, tree, node, child), i)
                   for i, child in enumerate(non_terminal_children))
    # zip something so I have a number

    return non_terminal_children[child]


def backpropagate_terminal(node: Node,
                           result: Result,
                           age: int):
    node.data.update_terminal_result(result, age)

    if node.is_root:
        return node

    node.parent.data.add_terminal_move(node.name)
    node = node.parent

    # The side to move can choose a win - they choose the quickest forcing line
    # This can only be entered if we have recursed at least once
    if same_side(result, node.data.board._player_to_move):
        age = min((c.data.board.age
                   for c in node.children
                   if c.name in node.data.non_terminal_moves))
        return backpropagate_terminal(node, result, age)
    # The parent only has continuations that lead to terminal results
    # None of them can be winning for this node, because otherwise we would
    # have previously set the terminal value of this node, and never visited it
    # again in the mcts
    elif not node.data.non_terminal_moves:
        # the parent will now choose a draw if possible
        # if forced to choose a loss, will choose the longest one
        # N.B. in connect4 all draws have age 42
        _, age, child = max((value_to_side(c.data.value,
                                           node.data.board._player_to_move),
                             c.data.board.age,
                             i)
                            for i, c in enumerate(node.children))
        result = node.children[child].data.board.result
        assert not same_side(result, node.data.board._player_to_move)
        return backpropagate_terminal(node, result, age)

    # nothing to do
    return node


def backpropagate(node: Node,
                  value: float):
    node.data.update_search_value(value)
    while not node.is_root:
        node = node.parent
        node.data.update_search_value(value)

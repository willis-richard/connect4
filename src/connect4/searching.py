import src.connect4.tree as t

from src.connect4.board import Board

from src.connect4.utils import Connect4Stats as info
from src.connect4.utils import same_side, Side, Result, result_to_side, value_to_side

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

    class PositionEvaluation():
        def __init__(self):
            self.value = None

        def update_value(self, value):
            self.value = value

        def __repr__(self):
            return "position_evaluation: " + str(self.value)

    class SearchEvaluation():
        def __init__(self):
            self.value = None

        def update_value(self, value):
            self.value = value

        def __repr__(self):
            return "tree_value: " + str(self.value)

    class NodeData(t.BaseNodeData):
        def __init__(self,
                     board: Board,
                     position_evaluation):#: GridSearch.PositionEvaluation):
            super().__init__(board,
                             position_evaluation,
                             GridSearch.SearchEvaluation())


class MCTS():
    class Config():
        def __init__(self, simulations, cpuct=None):
            self.simulations = simulations
            self.pb_c_base = 0
            self.pb_c_init = 0
            self.cpuct = 3.4 if cpuct is None else cpuct

    def __init__(self, config: Config):
        self.config = config

    def get_search_fn(self):
        return partial(mcts_search,
                       config=self.config,
                       evaluate_fn=evaluate_nn)

    class PositionEvaluation():
        def __init__(self):
            self.value = None
            self.policy_logits = None

        def update_value(self, value):
            self.value, self.policy_logits = value

        def __repr__(self):
            return "position_value: " + str(self.value) + \
                ", policy_logits: " + str(self.policy_logits)

    class SearchEvaluation():
        def __init__(self):
            # Although this is known at construction, I don't think the MCTS
            # wants to know this ahead of time
            self.terminal_value = None
            self.value_sum = 0.0
            self.visit_count = 0

        def update_value(self, value: float):
            self.value_sum += value
            self.visit_count += 1

        @property
        def value(self):
            if self.terminal_value is not None:
                return self.terminal_value
            if self.visit_count == 0:
                return 0
            return self.value_sum / self.visit_count

        def __repr__(self):
            return "terminal_value: " + str(self.terminal_value) + \
                ",  value_sum: " + str(self.value_sum) + \
                ",  visit_count: " + str(self.visit_count)

    class NodeData(t.BaseNodeData):
        def __init__(self,
                     board: Board,
                     position_evaluation):#: MCTS.PositionEvaluation):
            super().__init__(board,
                             position_evaluation,
                             MCTS.SearchEvaluation())
            self.non_terminal_moves = self.valid_moves.copy()

        def add_terminal_move(self, move):
            self.non_terminal_moves.remove(move)

        def __repr__(self):
            return super().__repr__() + \
                ",  non_terminal_moves: " + str(self.non_terminal_moves)


def evaluate_position_centre(node: Node):
    # return np.sum(np.multiply(board.o_pieces, info.value_grid) -
    # np.multiply(board.x_pieces, info.value_grid)) \
    # / float(info.value_grid_sum)
    board = node.data.board
    return 0.5 + \
        (np.einsum('ij,ij', board.o_pieces, info.value_grid)
         - np.einsum('ij,ij', board.x_pieces, info.value_grid)) \
        / float(info.value_grid_sum)


def grid_search(tree: t.Tree,
                board: Board,
                side: Side,
                plies: int,
                evaluate_fn: Callable[[Node], float]):
    tree.expand_node(tree.root, plies)
    nega_max(tree.root, plies, side, evaluate_fn)

    moves = np.array([(n.name) for n in tree.root.children])
    values = np.array([n.data.value for n in tree.root.children])
    idx = np.argmax(values) if side == Side.o else np.argmin(values)
    best_move_value = values[idx]

    best_move = moves[idx]
    return best_move, best_move_value


def nega_max(node: Node,
             plies: int,
             side: Side,
             evaluate_fn: Callable[[Node], float]):
    # https://en.wikipedia.org/wiki/Negamax
    if node.data.board.result is not None:
        node.data.update_position_value(node.data.board.result.value)
        return node.data.position_evaluation.value
    if plies == 0:
        node.data.update_position_value(evaluate_fn(node))
        return node.data.position_evaluation.value

    if side == Side.o:
        value = -2
        for child in node.children:
            value = max(value,
                        nega_max(child, plies - 1, Side.x, evaluate_fn))
    else:
        value = 2
        for child in node.children:
            value = min(value,
                        nega_max(child, plies - 1, Side.o, evaluate_fn))

    node.data.update_search_value(value)
    return value


def mcts_search(config: MCTS.Config,
                evaluate_fn: Callable[[MCTS.Config, Node],
                                      Tuple[float, Dict[int, float]]],
                tree: t.Tree,
                board: Board,
                side: Side):
    for _ in range(config.simulations):
        node = tree.root

        if not tree.root.data.non_terminal_moves:
            break

        # FIXME: node.data.evaluated() ?
        while node.children:
            node = select_child(config, tree, node)

        if node.data.board.result is not None:
            node = backpropagate_terminal(node,
                                          node.data.board.result,
                                          side)
            continue

        # position may be a in the transpositions table
        if node.data.position_evaluation.value is None:
            value, prior = evaluate_fn(config, node)
            value = value_to_side(value, side)
            prior = normalise_prior(node.data.valid_moves, prior)
            node.data.update_position_value((value, prior))
            node.data.update_search_value(value)
        value = node.data.position_evaluation.value
        tree.expand_node(node, 1)

        backpropagate(node, value)

    return select_action_not_stupid(config, tree)


def evaluate_nn(config: MCTS.Config, node: Node):
    # FIXME: remove when sure positions aren't evaluated multiple times
    assert node.data.position_evaluation.value is None
    value = evaluate_position_centre(node)
    policy_logits = {a: 1.0 + (1.0 / info.width) / (1 + np.abs(((info.width - 1) / 2.0) - a))
                     for a in range(info.width)}
    return value, policy_logits


def ucb_score(config: MCTS.Config, node: Node, child: Node):
    # pb_c = math.log((node.visit_count + config.pb_c_base + 1) /
    #                 config.pb_c_base) + config.pb_c_init
    # pb_c *= math.sqrt(node.visit_count) / (child.visit_count + 1)
    # pb_c = config.cpuct
    pb_c = config.cpuct * math.sqrt(node.data.search_evaluation.visit_count) \
           / (child.data.search_evaluation.visit_count + 1)

    prior_score = pb_c * node.data.position_evaluation.policy_logits[child.name]
    value_score = child.data.value
    return prior_score + value_score


def normalise_prior(valid_moves, policy_logits):
    policy = {a: math.exp(policy_logits[a])
              for a in valid_moves}
    policy_sum = sum(policy.values())
    new_policy = {}
    for action, p in policy.items():
        new_policy[action] = p / policy_sum
    return new_policy


def select_child(config: MCTS.Config,
                 tree: t.Tree,
                 node: Node):
    non_terminal_children = [child for child in node.children if
                             child.name in node.data.non_terminal_moves]
    _, child_name = max((ucb_score(config, node, child), child.name)
                        for child in non_terminal_children)
    child = node.data.child_map[child_name]

    return child


def select_action(config: MCTS.Config,
                  tree: t.Tree):
    # FIXME: removed softmax thing (would check game move history)
    _, action = max(((c.data.evaluation.visit_count, c.name)
                    for c in tree.root.children))

    # FIXME: what is the value of that child? avg value_sum?
    return action, tree.root.data.child_map[action].data.value


def select_action_not_stupid(config: MCTS.Config,
                             tree: t.Tree):
    print("ROOT:  ", tree.root)
    # FIXME: removed softmax thing (would check game move history)
    _, action = max(((c.data.value, c.name)
                    for c in tree.root.children))

    # FIXME: what is the value of that child? avg value_sum?
    return action, tree.root.data.child_map[action].data.value


def backpropagate_terminal(node: Node,
                           result: Result,
                           side: Side):
    node.data.search_evaluation.terminal_value = result_to_side(result, side)

    if node.is_root:
        return node

    node.parent.data.add_terminal_move(node.name)
    node = node.parent

    # The side to move can choose a win - assume that they will
    if same_side(result, node.data.board._player_to_move):
        return backpropagate_terminal(node, result, side)
    # The parent has only continuations that lead to terminal results
    elif not node.data.non_terminal_moves:
        # the parent will now choose a draw if possible
        choices = [c.data.search_evaluation.terminal_value for c in node.children]
        result = Result.draw if 0.5 in choices else Result(1 - node.data.board._player_to_move)
        assert not same_side(result, node.data.board._player_to_move)
        return backpropagate_terminal(node, result, side)

    # nothing to do
    return node


def backpropagate(node: Node,
                  value: float):
    while not node.is_root:
        node = node.parent
        node.data.update_search_value(value)

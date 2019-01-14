from src.connect4.board import Board
from src.connect4.tree import Tree

from src.connect4.utils import Connect4Stats as info
from src.connect4.utils import same_side, Side, Result

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
            self.cpuct = 3.4 if cpuct is None else cpuct

    def __init__(self, config: Config):
        self.config = config

    def get_search_fn(self):
        return partial(mcts_search,
                       config=self.config,
                       evaluate_fn=partial(evaluate_nn, config=self.config))

    class Evaluation():
        def __init__(self):
            self.value_sum = 0.0
            self.visit_count = 0
            self.policy_logits = None

        def evaluated(self):
            return self.visit_count != 0

        def add_value(self, value: float):
            self.value_sum += value
            self.visit_count += 1

        @property
        def value(self):
            if self.visit_count != 0:
                return self.value_sum / self.visit_count
            return 0.0

        def __repr__(self):
            return "value_sum: " + str(self.value_sum) + \
                ",  visit_count: " + str(self.visit_count) + \
                ", policy_logits: " + str(self.policy_logits)


def evaluate_position_centre(node: Node):
    # return np.sum(np.multiply(board.o_pieces, info.value_grid) -
    # np.multiply(board.x_pieces, info.value_grid)) \
    # / float(info.value_grid_sum)
    board = node.data.board
    return 0.5 + \
        (np.einsum('ij,ij', board.o_pieces, info.value_grid)
         - np.einsum('ij,ij', board.x_pieces, info.value_grid)) \
        / float(info.value_grid_sum)


def grid_search(tree: Tree,
                board: Board,
                side: Side,
                plies: int,
                evaluate_fn: Callable[[Node], float]):
    expand_tree(tree, tree.root, plies)
    nega_max(tree.root, plies, side, evaluate_fn)

    moves = np.array([(n.name) for n in tree.root.children])
    values = np.array([(n.data.get_value(side))
                       for n in tree.root.children])
    if side == Side.o:
        idx = np.argmax(values)
    else:
        idx = np.argmin(values)
    best_move_value = values[idx]

    best_move = moves[idx]
    return best_move, best_move_value


def expand_tree(tree: Tree,
                node: Node,
                plies):
    if plies == 0 or node.data.board.result is not None:
        return

    if not node.children:
        for move in node.data.valid_moves:
            tree.create_child(move, node)
        node.data.set_child_map(node.children)

    for child in node.children:
        expand_tree(tree, child, plies - 1)


def nega_max(node: Node,
             plies: int,
             side: Side,
             evaluate_fn: Callable[[Node], float]):
    # https://en.wikipedia.org/wiki/Negamax
    if node.data.board.result is not None:
        return node.data.board.result.value
    if plies == 0:
        value = evaluate_fn(node)
        node.data.evaluation.position_evaluation = value
        return value

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

    node.data.evaluation.tree_value = value
    return value


def mcts_search(config: MCTS.Config,
                evaluate_fn: Callable[[Node], Tuple[float, Dict[int, float]]],
                tree: Tree,
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
            value = node.data.board.result.value
            value = value if side == Side.o else (1 - value)
            node = backpropagate_terminal(node, node.data.board.result, value)
        else:
            value, policy_logits = evaluate_nn(config, node)
            value = value if side == Side.o else (1 - value)
            node.data.evaluation.policy_logits = policy_logits
            set_child_priors(tree, node)
        node.data.evaluation.add_value(value)

        backpropagate(node, value)

    return select_action_not_stupid(config, tree)


def evaluate_nn(config: MCTS.Config, node: Node):
    # FIXME: remove when sure positions aren't evaluated multiple times
    assert not node.data.evaluated()
    # FIXME: implement. N.B. google one is just some the expanding of the
    # children
    value = evaluate_position_centre(node)
    # print ("posn value: ", value)
    policy_logits = {a: 1.0 + (1.0 / info.width) / (1 + np.abs(((info.width - 1) / 2.0) - a))
                     for a in range(info.width)}
    return value, policy_logits


def ucb_score(config: MCTS.Config, node: Node, child: Node):
    # pb_c = math.log((node.visit_count + config.pb_c_base + 1) /
    #                 config.pb_c_base) + config.pb_c_init
    # pb_c *= math.sqrt(node.visit_count) / (child.visit_count + 1)
    # pb_c = config.cpuct
    pb_c = config.cpuct * math.sqrt(node.data.evaluation.visit_count) \
           / (child.data.evaluation.visit_count + 1)

    prior_score = pb_c * child.data.evaluation.prior
    value_score = child.data.evaluation.value
    # print("child, prior, value, total:  ", child.name, prior_score, value_score, prior_score + value_score)
    return prior_score + value_score


def select_child(config: MCTS.Config, tree: Tree, node: Node):
    non_terminal_children = [child for child in node.children if
                             child.name in node.data.non_terminal_moves]
    if not non_terminal_children:
        print("Argh:\nparent:\n", node.parent, "\nnode:\n", node, "\nchildren\n", node.children)
    # print([[ucb_score(config, node, child), child.name] for child in non_terminal_children])
    _, child_name = max((ucb_score(config, node, child), child.name)
                        for child in non_terminal_children)
    child = node.data.child_map[child_name]

    return child


def set_child_priors(tree: Tree, node: Node):
    expand_tree(tree, node, 1)
    policy = {a: math.exp(node.data.evaluation.policy_logits[a])
              for a in node.data.valid_moves}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.data.child_map[action].data.evaluation.prior = p / policy_sum
        # NAH JUST UPDATE IT@S POLICY LOGITS


def select_action(config: MCTS.Config, tree: Tree):
    # FIXME: removed softmax thing (would check game move history)
    _, action = max(((c.data.evaluation.visit_count, c.name)
                    for c in tree.root.children))

    # FIXME: what is the value of that child? avg value_sum?
    return action, tree.root.data.child_map[action].data.get_value(tree.root.data.board._player_to_move)


def select_action_not_stupid(config: MCTS.Config, tree: Tree):
    # FIXME: removed softmax thing (would check game move history)
    _, action = max(((c.data.evaluation.value, c.name)
                    for c in tree.root.children))

    # FIXME: what is the value of that child? avg value_sum?
    return action, tree.root.data.child_map[action].data.get_value(tree.root.data.board._player_to_move)


def backpropagate_terminal(node: Node,
                           result: Result,
                           value: float):
    node.data.terminal_value = value

    if node.is_root:
        return node
    node.parent.data.add_terminal_move(node.name)

    # The side who just moved can choose a win - assume that they will
    if same_side(result, Side(1 - node.data.board._player_to_move)):
        return backpropagate_terminal(node.parent, result, value)
    # The parent has only continuations that lead to terminal results
    elif len(node.parent.data.non_terminal_moves) == 0:
        # the parent will now choose a draw if possible
        choices = [c.data.terminal_value for c in node.parent.children]
        value = 0.5 if 0.5 in choices else value
        result = Result.draw if 0.5 in choices else result
        return backpropagate_terminal(node.parent, result, value)

    # nothing to do
    return node


def backpropagate(node: Node,
                  value: float):
    while not node.is_root:
        node = node.parent
        # FIXME: to confirm
        node.data.evaluation.add_value(value)

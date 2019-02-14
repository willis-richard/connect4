from src.connect4.board import Board
from src.connect4.evaluators import Evaluator
from src.connect4.player import BasePlayer
from src.connect4.tree import BaseNodeData, Tree
from src.connect4.utils import (same_side,
                                Side,
                                value_to_side,
                                Result)

from anytree import Node
import math
from typing import Callable, List, Set, Tuple


class MCTSConfig():
    def __init__(self, simulations, cpuct=9999):
        self.simulations = simulations
        self.pb_c_base = 0
        self.pb_c_init = 0
        self.cpuct = cpuct


class PositionEvaluation():
    def __init__(self,
                 value: float,
                 prior: List[float]):
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


class NodeData(BaseNodeData):
    def __init__(self, board: Board):
        super().__init__(board)
        self.terminal_result = None
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
            return value_to_side(float(self.terminal_result[0].value),
                                 side)
        return super().value(side)

    def __str__(self):
        # if self.position_value is not None:
        #     return "prior: " + str(self.position_value.prior) + \
        #         "   terminal_result: " + str(self.terminal_result) + \
        #         "  " + super().__str__()
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
                    self.evaluator.result_table,
                    NodeData)

        search(self.config, tree, self.evaluator)

        if tree.root.data.terminal_result is None:
            move, value = tree.select_best_move()
        # choose shortest win
        elif same_side(tree.root.data.terminal_result, tree.side):
            print("Yay")
            _, move, value = min((c.data.terminal_result[1],
                                  c.name,
                                  tree.get_node_value(c))
                                 for c in tree.root.children
                                 if c.name in tree.root.data.terminal_moves)
        # else longest loss (or draw = 42)
        else:
            print("Crap")
            _, move, value = max((c.data.terminal_result[1],
                                  c.name,
                                  tree.get_node_value(c))
                                 for c in tree.root.children
                                 if c.name in tree.root.data.terminal_moves)
        board.make_move(move)
        return move, value, tree

    def __str__(self):
        return super().__str__() + ", type: Computer"


def search(config: MCTSConfig,
           tree: Tree,
           evaluator: Callable[[Board],
                               Tuple[float, List[float]]]):
    for _ in range(config.simulations):
        node = tree.root

        if not tree.root.data.non_terminal_moves:
            break

        # FIXME: node.data.evaluated() ?
        while node.children:
            node = select_child(config, tree, node)

        # previously evaluated, so expand
        if node.data.position_value is not None:
            tree.expand_node(node, 1)
            node = select_child(config, tree, node)

        board = node.data.board
        # encountered a new terminal position
        if board.result is not None:
            value = board.result.value
            node.data.position_value = PositionEvaluation(value, [])
            node = backpropagate_terminal(node,
                                          board.result,
                                          board.age)
        else:
            value, prior = evaluator(board)
            node.data.position_value = PositionEvaluation(value, prior)

        backpropagate(node, value)

    return


def select_child(config: MCTSConfig,
                 tree: Tree,
                 node: Node):
    non_terminal_children = [child for child in node.children if
                             child.name not in node.data.terminal_moves]
    _, child = max((ucb_score(config, tree, node, child), i)
                   for i, child in enumerate(non_terminal_children))

    return non_terminal_children[child]


def ucb_score(config: MCTSConfig,
              tree: Tree,
              node: Node,
              child: Node):
    # pb_c = math.log((node.visit_count + config.pb_c_base + 1) /
    #                 config.pb_c_base) + config.pb_c_init
    # pb_c *= math.sqrt(node.visit_count) / (child.visit_count + 1)
    # pb_c = config.cpuct
    pb_c = config.cpuct * (
        math.sqrt(node.data._search_value.visit_count)
        / (child.data._search_value.visit_count + 1))

    prior_score = pb_c * node.data.position_value.prior[child.name]
    value_score = tree.get_node_value(child)
    return prior_score + value_score


def backpropagate_terminal(node: Node,
                           result: Result,
                           age: int):
    node.data.terminal_result = (result, age)

    if node.is_root:
        return node

    node.parent.data.terminal_moves.add(node.name)
    node = node.parent

    # The side to move can choose a win - they choose the quickest forcing line
    # This can only be entered if we have recursed at least once
    if same_side(result, node.data.board._player_to_move):
        age = min((c.data.board.age
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
        _, age, child = max((value_to_side(
            c.data.terminal_result[0].value,
            node.data.board._player_to_move),
                             c.data.board.age,
                             i)
                            for i, c in enumerate(node.children))
        result = node.children[child].data.terminal_result[0]
        assert not same_side(result, node.data.board._player_to_move)
        return backpropagate_terminal(node, result, age)

    # nothing to do
    return node


def backpropagate(node: Node,
                  value: float):
    node.data._search_value.add(value)
    while not node.is_root:
        node = node.parent
        node.data._search_value.add(value)

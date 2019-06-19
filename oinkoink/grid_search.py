from oinkoink.board import Board
from oinkoink.evaluators import Evaluator
from oinkoink.player import BasePlayer
from oinkoink.tree import Tree
from oinkoink.utils import same_side, Side

from anytree import Node

from typing import Callable, Dict, Tuple


class GridSearch(BasePlayer):
    def __init__(self,
                 name: str,
                 plies: int,
                 evaluator: Evaluator):
        super().__init__(name)
        self.plies = plies
        self.evaluator = evaluator

    def make_move(self, board):
        tree = Tree(board)

        tree.expand_node(tree.root,
                         self.plies)
        nega_max(tree.root,
                 self.plies,
                 self.evaluator)

        child = tree.best_move()
        board.make_move(child.name)
        return child.name, child.data.absolute_value, tree

    def __str__(self):
        return super().__str__() + ", type: Computer"


def nega_max(node: Node,
             plies: int,
             evaluator: Callable[[Board],
                                 Tuple[float, Dict[int, float]]]):
    # https://en.wikipedia.org/wiki/Negamax
    if node.data.board.result is not None:
        board = node.data.board
        # Prefer faster wins and slower losses
        if same_side(board.result, Side.o):
            value = board.result.value - board.age / 10000.0
        else:
            value = board.result.value + board.age / 10000.0
        node.data.position_value = value
        return value
    if plies == 0:
        value = evaluator(node.data.board)
        node.data.position_value = value
        return value

    side = node.data.board.player_to_move

    if side == Side.o:
        value = -2
        for child in node.children:
            value = max(value,
                        nega_max(child, plies - 1, evaluator))
    else:
        value = 2
        for child in node.children:
            value = min(value,
                        nega_max(child, plies - 1, evaluator))

    node.data.search_value = value
    return value

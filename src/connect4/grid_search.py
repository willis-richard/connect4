from src.connect4.player import BasePlayer
from src.connect4.tree import BaseNodeData
from src.connect4.utils import Side

from anytree import Node


class GridSearch(BasePlayer):
    def __init__(self,
                 name: str
                 config,
                 evaluator):
        super().__init__(name)
        self.config = config
        self.evaluator = evaluator

    class NodeData(BaseNodeData):
        def __init__(self):
            return

    def make_move(self, board):
        self.tree = Tree(board, self.evaluator.transition_t)

        self.tree.expand_node(self.tree.root,
                              self.config.plies)
        self.nega_max(self.tree.root,
                      self.config.plies)

        move, value = self.tree.select_best_move()
        board.make_move(move)
        return move, value

    def nega_max(node: Node,
                 plies: int):
        # https://en.wikipedia.org/wiki/Negamax
        if node.data.board.result is not None:
            # Prefer faster wins
            board = node.data.board
            return board.result.value - board.age / 1000.0
        if plies == 0:
            value, _ = self.evaluator(node)
            node.data.value = value
            return value

        side = node.data.board._player_to_move

        if side == Side.o:
            value = -2
            for child in node.children:
                value = max(value,
                            nega_max(child, plies - 1))
            else:
                value = 2
                for child in node.children:
                    value = min(value,
                                nega_max(child, plies - 1))

        node.data.value = value
        return value

    def __str__(self):
        return super().__str__() + ", type: Computer"

from connect4.board_c import Board
import connect4.evaluators as ev
from connect4.mcts import MCTS, MCTSConfig

from connect4.neural.config import ModelConfig
from connect4.neural.nn_pytorch import ModelWrapper

import anytree
from functools import partial
import numpy as np
import sys


if __name__ == "__main__":
    np.set_printoptions(5)
    array = np.genfromtxt(sys.argv[1], dtype='c')
    o_pieces = np.zeros(array.shape, dtype=np.bool_)
    x_pieces = np.zeros(array.shape, dtype=np.bool_)
    o_pieces[np.where(array == b'o')] = 1
    x_pieces[np.where(array == b'x')] = 1

    board = Board.from_pieces(o_pieces, x_pieces)

    model_config = ModelConfig()
    model = ModelWrapper(model_config, file_name=sys.argv[2])
    value, prior = model(board)
    print('{}\nvalue {}, policy {}'.format(board, value, prior))

    player = MCTS("name",
                  MCTSConfig(simulations=8),
                  ev.Evaluator(partial(ev.evaluate_nn,
                                       model=model)))

    move, value, tree = player.make_move(board)
    prior = tree.get_policy()
    print("move {}, value {:4f}, policy {}".format(move, value, prior))

    for r_child in tree.root.children:
        for child in r_child.children:
            child.children = []

    for pre, fill, node in anytree.RenderTree(tree.root):
        print("{}{}, {:.3f} ({}, {}, {}), {}, {}".format(
            pre,
            node.name,
            tree.get_node_value(node),
            node.data.board.result,
            node.data.search_value,
            node.data.position_value,
            node.data.search_value.visit_count if node.data.search_value else 0,
            node.data.position_value.prior if node.data.position_value else None))

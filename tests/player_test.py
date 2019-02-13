from src.connect4.board import Board
import src.connect4.evaluators as evaluators
from src.connect4.grid_search import GridSearch
from src.connect4.mcts import MCTS, MCTSConfig

import anytree
import pytest

import numpy as np

from copy import copy

o_pieces = [
    np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 0],
         [1, 0, 0, 1, 0, 0, 0]], dtype=np.bool_),
    np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 0],
         [1, 0, 0, 1, 0, 0, 0]], dtype=np.bool_),
    np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0]], dtype=np.bool_),
    np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0],
         [0, 0, 0, 1, 0, 0, 1],
         [0, 0, 1, 0, 1, 1, 1],
         [0, 0, 1, 0, 1, 1, 1]], dtype=np.bool_),
    np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0]], dtype=np.bool_)

]

x_pieces = [
    np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 1, 0, 1, 0, 0]], dtype=np.bool_),
    np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 0]], dtype=np.bool_),
    np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0]], dtype=np.bool_),
    np.array(
        [[0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 1, 0, 0, 0]], dtype=np.bool_),
    np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]], dtype=np.bool_)
]

plies = [1, 2, 2, 4, 4]
ans = [[1], [1], [1], [6], [2, 5]]

assert len(o_pieces) == len(x_pieces) == len(plies) == len(ans)

boards = [Board(o_pieces=o, x_pieces=x)
          for o, x in zip(o_pieces, x_pieces)]


@pytest.mark.parametrize("n,board,plies,ans",
                         [(n, b, d, a) for n, b, d, a
                          in zip(range(len(ans)), boards, plies, ans)])
def test_next_move(n, board, plies, ans):
    computers = [
        # GridSearch("grid_test",
        #                     plies,
        #                     evaluators.Evaluator(
        #                         evaluators.evaluate_centre)),
                 MCTS("mcts_test",
                      MCTSConfig(simulations=7**plies + 1,
                                 cpuct=9999),
                      evaluators.Evaluator(
                          evaluators.evaluate_centre_with_prior))]
    for computer in computers:
        print(board)

        move, _, tree = computer.make_move(board)

        if plies <= 2:
            for pre, fill, node in anytree.RenderTree(tree.root):
                print("%s%s, %3d, %s, %s, %s" % (
                    pre,
                    node.name,
                    tree.get_node_value(node),
                    node.data.board.result,
                    node.data.position_value,
                    node.data.search_value))

        assert move in ans
    return


def test_multiple_moves():
    board = Board()

    computer = GridSearch("grid_test",
                          4,
                          evaluators.Evaluator(
                              evaluators.evaluate_centre))
    board.make_move(3)

    assert computer.make_move(board)[0] == 3

    board.make_move(3)

    assert computer.make_move(board)[0] == 3

    board.make_move(4)

    assert computer.make_move(board)[0] == 2

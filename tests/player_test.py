from oinkoink.board import Board
import oinkoink.evaluators as evaluators
from oinkoink.grid_search import GridSearch
from oinkoink.mcts import MCTS, MCTSConfig

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
         [0, 0, 0, 1, 1, 0, 0]], dtype=np.bool_),
    np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 1, 0, 0],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 0]], dtype=np.bool_),
    np.array(
        [[0, 0, 0, 0, 1, 1, 1],
         [0, 0, 1, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 0],
         [0, 0, 0, 1, 1, 1, 0],
         [0, 0, 1, 0, 0, 0, 1],
         [1, 0, 0, 1, 1, 1, 0]], dtype=np.bool_)
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
         [0, 0, 0, 0, 0, 0, 0]], dtype=np.bool_),
    np.array(
        [[0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0],
         [0, 1, 1, 0, 0, 0, 1]], dtype=np.bool_),
    np.array(
        [[0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1],
         [0, 0, 0, 1, 1, 0, 1],
         [0, 0, 1, 0, 0, 0, 1],
         [0, 0, 0, 1, 1, 1, 0],
         [0, 0, 1, 0, 0, 0, 1]], dtype=np.bool_)
]

plies = [1, 2, 2, 4, 4, 2, 15]
ans = [[1], [1], [1], [6], [2, 5], [4], [0]]

assert len(o_pieces) == len(x_pieces) == len(plies) == len(ans)

boards = [Board.from_pieces(o_pieces=o, x_pieces=x)
          for o, x in zip(o_pieces, x_pieces)]


@pytest.mark.parametrize("n,board,plies,ans",
                         [(n, b, d, a) for n, b, d, a
                          in zip(range(len(ans)), boards, plies, ans)])
def test_grid_next_move(n, board, plies, ans):
    computer = GridSearch("grid_test",
                          plies,
                          evaluators.Evaluator(
                              evaluators.evaluate_centre))
    board_copy = copy(board)
    print(board)

    move, _, tree = computer.make_move(board_copy)

    if plies <= 2:
        for pre, fill, node in anytree.RenderTree(tree.root):
            # print("%s%s, %3d, %s, %s, %s" % (
            print("%s%s, %s" % (
                pre,
                node.name,
                node.data))

    assert move in ans
    return


@pytest.mark.parametrize("n,board,plies,ans",
                         [(n, b, d, a) for n, b, d, a
                          in zip(range(len(ans)), boards, plies, ans)])
def test_mcts_next_move(n, board, plies, ans):
    computer = MCTS("mcts_test",
                    MCTSConfig(simulations=7**plies + 1 if plies <= 6 else 2**plies,
                               pb_c_init=9999),
                    evaluators.Evaluator(
                        evaluators.evaluate_centre_with_prior))
    board_copy = copy(board)
    print(board)

    move, _, tree = computer.make_move(board_copy)

    for r_child in tree.root.children:
        for child in r_child.children:
            child.children = []
    for pre, fill, node in anytree.RenderTree(tree.root):
        print("%s%s, %3d, %s, %s, %s, %s" % (
            pre,
            node.name,
            tree.get_node_value(node),
            node.data.board.result,
            node.data.position_value,
            node.data.search_value,
            0 if node.data.search_value is None else node.data.search_value.visit_count))

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

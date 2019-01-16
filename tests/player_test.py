from src.connect4.utils import Side
from src.connect4 import board
from src.connect4 import player
from src.connect4 import searching

import anytree
import pytest

import numpy as np

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


@pytest.mark.parametrize("n,o_pieces,x_pieces,plies,ans",
                         [(n, o, x, d, a) for n, o, x, d, a in zip(range(len(ans)), o_pieces, x_pieces, plies, ans)])
def test_next_move(n, o_pieces, x_pieces, plies, ans):
    board_ = board.Board(o_pieces=o_pieces,
                         x_pieces=x_pieces)

    computer = player.ComputerPlayer("test_name",
                                     searching.GridSearch(plies=4))
    computer.side = board_._player_to_move

    print(board_)

    move = computer.make_move(board_)

    if plies <= 2:
        for pre, fill, node in anytree.RenderTree(computer.tree.root):
            print("%s%s, %s, %s, %s, %s" % (pre,
                node.name,
                node.data.board_result,
                node.data.position_evaluation,
                node.data.search_evaluation,
                node.data.get_value(Side.x)))


    assert move in ans
    return


def test_multiple_moves():
    board_ = board.Board()

    computer = player.ComputerPlayer("test_name",
                                     searching.GridSearch(plies=4))
    computer.side = Side.x

    board_.make_move(3)

    assert computer.make_move(board_) == 3

    board_.make_move(3)

    assert computer.make_move(board_) == 3

    board_.make_move(4)

    assert computer.make_move(board_) == 2

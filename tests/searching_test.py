from src.connect4.board import Board
from src.connect4.utils import Side
from src.connect4.player import ComputerPlayer
from src.connect4.searching import MCTS

import anytree
import numpy as np


#def test_mcts():
#    board_ = Board()
#
#    computer = ComputerPlayer("test_name",
#                              MCTS(MCTS.Config(simulations=8)))
#
#    # assert computer.make_move(board_) == 0
#    computer.make_move(board_)
#
#    for pre, fill, node in anytree.RenderTree(computer.tree.root):
#            print("%s%s, %s, %s, %s, %s" % (pre,
#                node.name,
#                node.data.board.result,
#                node.data.position_evaluation,
#                node.data.search_evaluation,
#                node.data.value))
#
#    assert False
#    # board_.make_move(3)
#
#    # assert computer.make_move(board_) == 3
#
#    # board_.make_move(3)
#
#    # assert computer.make_move(board_) == 3
#
#    # board_.make_move(4)
#
#    # assert computer.make_move(board_) == 2

def test_single():
    o_pieces = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 0],
         [1, 0, 0, 1, 0, 0, 0]], dtype=np.bool_)
    x_pieces = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 1, 0, 1, 0, 0]], dtype=np.bool_)

    board = Board(o_pieces=o_pieces,
                  x_pieces=x_pieces)

    computer = ComputerPlayer("test_name",
                              MCTS(MCTS.Config(simulations=8, cpuct=9999)))

    print(board)

    move, _ = computer.make_move(board)

    for pre, fill, node in anytree.RenderTree(computer.tree.root):
            print("%s%s, %s, %s, %s, Node value: %s" % (pre,
                node.name,
                node.data.board.result,
                node.data.position_evaluation,
                node.data.search_evaluation,
                node.data.value))

    assert move in [1]

from src.connect4.board import Board
from src.connect4.utils import Side
from src.connect4.player import ComputerPlayer
from src.connect4.searching import MCTS

import anytree


def test_mcts():
    board_ = Board()

    computer = ComputerPlayer("test_name",
                              MCTS(MCTS.Config(simulations=4)))
    computer.side = Side.o

    # assert computer.make_move(board_) == 0
    computer.make_move(board_)

    for pre, fill, node in anytree.RenderTree(computer.tree.root):
            print("%s%s, %s, %s, %s, %s" % (pre,
                node.name,
                node.data.board_result,
                node.data.position_evaluation,
                node.data.search_evaluation,
                node.data.get_value(computer.side)))

    assert False
    # board_.make_move(3)

    # assert computer.make_move(board_) == 3

    # board_.make_move(3)

    # assert computer.make_move(board_) == 3

    # board_.make_move(4)

    # assert computer.make_move(board_) == 2

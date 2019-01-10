from src.connect4.board import Board
from src.connect4.player import ComputerPlayer
# from src.connect4 import transition_table

from functools import partial


def test_transition_table():
    computer_1 = ComputerPlayer("test_name",
                                partial(ComputerPlayer.grid_search,
                                        plies=2))
    computer_1.side = 1

    board = Board()

    computer_1.make_move(board)

    print(computer_1.tree.transition_t)

    assert(len(computer_1.tree.transition_t.entries) == 3)
    assert(len(computer_1.tree.transition_t.entries[2].table) == 49)

    board.make_move(3)

    computer_1.tree.update_root(board)

    print(computer_1.tree.transition_t)

    assert(len(computer_1.tree.transition_t.entries) == 2)

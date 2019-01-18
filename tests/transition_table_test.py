from src.connect4.board import Board
from src.connect4.player import ComputerPlayer
from src.connect4.searching import GridSearch
# from src.connect4 import transition_table


def disabled_test_transition_table():
    computer_1 = ComputerPlayer("test_name",
                                GridSearch(plies=2))
    board = Board()

    computer_1.make_move(board)

    print(computer_1.tree.transition_t)

    assert(len(computer_1.tree.transition_t.entries) == 3)
    assert(len(computer_1.tree.transition_t.entries[2].table) == 49)

    board.make_move(3)

    computer_1.tree.update_root(board)

    print(computer_1.tree.transition_t)

    assert(len(computer_1.tree.transition_t.entries) == 2)

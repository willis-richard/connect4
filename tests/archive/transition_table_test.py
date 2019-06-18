from connect4.board import Board
import connect4.evaluators as evaluators
from connect4.grid_search import GridSearch
# from connect4 import transition_table


def disabled_test_transition_table():
    computer_1 = GridSearch("grid_1",
                            2,
                            evaluators.Evaluator(
                                evaluators.evaluate_centre))
    board = Board()

    computer_1.make_move(board)

    print(computer_1.tree.transition_t)

    assert(len(computer_1.tree.transition_t.entries) == 3)
    assert(len(computer_1.tree.transition_t.entries[2].table) == 49)

    board.make_move(3)

    computer_1.tree.update_root(board)

    print(computer_1.tree.transition_t)

    assert(len(computer_1.tree.transition_t.entries) == 2)

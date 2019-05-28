from connect4.board_c import Board
import connect4.evaluators as ev
from connect4.grid_search import GridSearch

if __name__ == '__main__':
    player = GridSearch("grid",
                        4,
                        ev.Evaluator(ev.evaluate_centre))

    for _ in range(int(1e4)):
        board = Board()
        player.make_move(board)

    # Add to leaky code within python_script_being_profiled.py
    from pympler import muppy, summary
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)

    # Prints out a summary of the large objects
    summary.print_(sum1)

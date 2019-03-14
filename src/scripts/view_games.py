from src.connect4.board import Board

import pickle
import sys


if __name__ == "__main__":
    with open(sys.argvp[1], 'rb') as f:
        games = pickle.loads(f)
        game = games[sys.argv[2]]

        board = Board()
        for move, value in game:
            board.make_move(move)
            print(board)
            print("Value = ", value)

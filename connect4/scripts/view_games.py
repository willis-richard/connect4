from connect4.board import Board

from connect4.neural.storage import game_str

import pickle
import sys


if __name__ == "__main__":
    print(sys.argv[1])
    with open(sys.argv[1], 'rb') as f:
        games = pickle.load(f)
        print('{} games loaded'.format(len(games)))
        game = games[int(sys.argv[2])]

        print(game_str(game.moves,
                       game.values,
                       game.priors))

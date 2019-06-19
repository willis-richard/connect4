from oinkoink.board import Board

from oinkoink.neural.storage import game_str

from collections import Counter
import pickle
import sys


if __name__ == "__main__":
    print(sys.argv[1])
    with open(sys.argv[1], 'rb') as f:
        games = pickle.load(f)
        print('{} games loaded'.format(len(games)))
        move_count = Counter(range(7))
        for game in games:
            move_count[game.moves[0]] += 1

        print(move_count)

import oinkoink.evaluators as ev
from oinkoink.grid_search import GridSearch
from oinkoink.utils import Side

from oinkoink.neural.pytorch.data import Connect4Dataset, native_to_pytorch

from oinkoink.scripts.view_boards import read_8ply_data

from copy import copy
import numpy as np
import pickle


def return_half(x):
    return 0.5


DATA_DIR = '/home/richard/data/connect4'

if __name__ == '__main__':
    boards_8ply, values_8ply = read_8ply_data(add_fliplr=True)
    # with open(DATA_DIR + '/8ply_boards.pkl', 'rb') as f:
    #     boards_8ply = pickle.load(f)
    # with open(DATA_DIR + '/8ply_values.pkl', 'rb') as f:
    #     values_8ply = pickle.load(f)
    table = {b: v for b, v in zip(boards_8ply, values_8ply)}

    # board_ips = make_random_ips(7)
    # with open(DATA_DIR + '/7ply_ips.pkl', 'wb') as f:
    #     pickle.dump(board_ips, f)

    with open(DATA_DIR + '/7ply_ips.pkl', 'rb') as f:
        board_ips = pickle.load(f)

    print("Number of 7ply ips: {}".format(len(board_ips)))
    print("len of table at start: {}".format(len(table)))
    ten_percent = int(len(board_ips) / 10)

    evaluator = ev.Evaluator(return_half)
    player_1 = GridSearch("1",
                          1,
                          evaluator)
    player_2 = GridSearch("2",
                          2,
                          evaluator)

    boards = []
    values = []
    priors = []
    unknown = []
    for i, board in enumerate(board_ips):
        if i % ten_percent == 0:
            print(i)
        to_move = board.player_to_move
        undetermined = False
        moves = np.zeros((7,))
        valid_moves = board.valid_moves
        for move in valid_moves:
            new_board = copy(board)
            new_board.make_move(move)
            value = table.get(new_board)
            if value is None:
                value = new_board.result
                if value is None:
                    # Now we look into whether the resulting 8 ply position has a trivial solution
                    _, value, _ = player_1.make_move(
                        copy(new_board))
                    if value == 0.5:
                        _, value, _ = player_2.make_move(
                            copy(new_board))
                        if value == 0.5:
                            unknown.append(board)
                            undetermined = True
                            break
                    value = float(int(np.round(value)))
                    table[new_board] = value
                    table[new_board.create_fliplr()] = value
                else:
                    value = new_board.result.value
            moves[move] = value
        if undetermined:
            continue
        value = np.max(moves) if to_move == Side.o else np.min(moves)
        # if there is no winning move, we want to set the prior to be any
        # legal move
        prior = np.array([1.0 if moves[x] == value and x in valid_moves
                          else 0.0
                          for x in range(7)])
        prior = prior / np.sum(prior) if \
            np.sum(prior) > 0.0 else \
            np.zeros((7,))
        boards.append(board)
        values.append(value)
        priors.append(prior)

    print("Finished: {} {} {}".format(len(boards), len(values), len(priors)))
    print("{} known 8ply non-terminal positions".format(len(table)))
    print("{} unknown (prior, not necessarily value) 7ply non-terminal positions".format(len(unknown)))
    with open(DATA_DIR + '/8ply_table.pkl', 'wb') as f:
        pickle.dump(table, f)
    with open(DATA_DIR + '/7ply_boards.pkl', 'wb') as f:
        pickle.dump(boards, f)
    with open(DATA_DIR + '/7ply_values.pkl', 'wb') as f:
        pickle.dump(values, f)
    with open(DATA_DIR + '/7ply_priors.pkl', 'wb') as f:
        pickle.dump(priors, f)
    with open(DATA_DIR + '/7ply_unknown_boards.pkl', 'wb') as f:
        pickle.dump(unknown, f)

    board_t, value_t, prior_t = native_to_pytorch(boards, values, priors)
    data_7ply = Connect4Dataset(board_t, value_t, prior_t)
    data_7ply.save(DATA_DIR + '/connect4dataset_7ply.pth')

    boards_8ply = []
    values_8ply = []
    for b, v in table.items():
        boards_8ply.append(b)
        values_8ply.append(v)
    print("len of boards 8ply: {}".format(len(boards_8ply)))

    with open(DATA_DIR + '/8ply_boards_extended.pkl', 'wb') as f:
        pickle.dump(boards_8ply, f)
    with open(DATA_DIR + '/8ply_values_extended.pkl', 'wb') as f:
        pickle.dump(values_8ply, f)

    board_t, value_t, _ = native_to_pytorch(boards_8ply, values_8ply)
    data_8ply = Connect4Dataset(board_t, value_t, _)
    data_8ply.save(DATA_DIR + '/connect4datasetextended_8ply.pth')

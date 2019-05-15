from connect4.board import make_random_ips
import connect4.evaluators as ev
from connect4.grid_search import GridSearch
from connect4.utils import Side

from connect4.scripts.view_boards import read_8ply_data

from copy import copy
import numpy as np
import pickle


def return_half(x):
    return 0.5


if __name__ == '__main__':
    # boards_8ply, values_8ply = read_8ply_data(False)
    # table = {b: v for b, v in zip(boards_8ply, values_8ply)}
    # table.update([(b.create_fliplr(), v) for b, v in zip(boards_8ply, values_8ply)])
    # with open('/home/richard/Downloads/8ply_table.pkl', 'wb') as f:
    #     pickle.dump(table, f)

    # board_ips = make_random_ips(7)
    # with open('/home/richard/Downloads/8ply_ips.pkl', 'wb') as f:
    #     pickle.dump(board_ips, f)

    with open('/home/richard/Downloads/8ply_table.pkl', 'rb') as f:
        table = pickle.load(f)
    with open('/home/richard/Downloads/8ply_ips.pkl', 'rb') as f:
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
    for i, board in enumerate(board_ips):
        if i % ten_percent == 0:
            print(i)
        to_move = board._player_to_move
        undetermined = False
        moves = np.zeros((7,))
        valid_moves = board.valid_moves
        for move in valid_moves:
            new_board = copy(board)
            new_board._make_move(move)
            value = table.get(new_board)
            if value is None:
                value = new_board.check_terminal_position()
                if value is None:
                    _, value, _ = player_1.make_move(
                        copy(new_board))
                    if value == 0.5:
                        _, value, _ = player_2.make_move(
                            copy(new_board))
                        if value == 0.5:
                            undetermined = True
                            break
                    value = np.round(value)
                    table[new_board] = value
                    table[new_board.fliplr()] = value
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
        boards.append(board)
        values.append(value)
        priors.append(prior)

    print("Finished: {} {} {}".format(len(boards), len(values), len(priors)))
    print("len of table at end: {}".format(len(table)))
    with open('/home/richard/Downloads/7ply_boards.pkl', 'wb') as f:
        pickle.dump(boards, f)
    with open('/home/richard/Downloads/7ply_values.pkl', 'wb') as f:
        pickle.dump(values, f)
    with open('/home/richard/Downloads/7ply_priors.pkl', 'wb') as f:
        pickle.dump(priors, f)
    # torch.save(torch.stack(all_boards),
    #            open('~/Downloads/7ply_boards.pth'))
    # torch.save(torch.stack(all_values),
    #            open('~/Downloads/7ply_values.pth'))
    # torch.save(torch.stack(all_priors),
    #            open('~/Downloads/7ply_priors.pth'))

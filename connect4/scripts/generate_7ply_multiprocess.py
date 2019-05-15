from connect4.board import make_random_ips
import connect4.evaluators as ev
from connect4.grid_search import GridSearch
from connect4.utils import Side, value_to_side

from connect4.scripts.view_boards import read_8ply_data

from copy import copy
from functools import partial
from multiprocessing import Manager, Pool
import numpy as np
import pickle
import torch


def backpropagate(board_ips, table, player):
    to_move = board_ips[0]._player_to_move
    boards = []
    values = []
    priors = []
    for j, board in enumerate(board_ips):
        moves = np.zeros((7,))
        max_value = value_to_side(0.0, to_move)
        valid_moves = board.valid_moves
        undetermined = False
        for move in valid_moves:
            new_board = copy(board)
            if new_board.make_move(move) is None:
                value = table.get(new_board)
                if value is None:
                    new_board_flip = new_board.create_fliplr()
                    value = table.get(new_board_flip)
                    if value is None:
                        # apparently the move is now forcing. Try to solve.
                        _, value, _ = player.make_move(
                            copy(new_board))

                        if value != 0.5:
                            value = np.round(value)
                            table[new_board] = value
                            table[new_board_flip] = value
                        else:
                            # We cannot determine the forcing move For now
                            # assume lost but we can recover the value of
                            # board if there are forcing wins for that
                            # player
                            value = value_to_side(0.0, to_move)
                            undetermined = True
                    else:
                        table[new_board] = value
                moves[move] = value
            else:
                moves[move] = new_board.result.value
        value = np.max(moves) if to_move == Side.o else np.min(moves)
        if undetermined:
            if value == value_to_side(1.0, to_move):
                # we can add the result of the board to the table, but we don't know the correct prior, so we don't add it to the training set
                table[board] = value
            continue
        # if there is no winning move, we want to set the prior to be any
        # legal move
        prior = [1 if moves[x] == value and x in valid_moves else 0
                 for x in range(7)]
        boards.append(board)
        values.append(value)
        priors.append(prior)
    return boards, values, priors


def return_half(x):
    return 0.5


if __name__ == '__main__':
    boards_8ply, values_8ply = read_8ply_data(False)

    mgr = Manager()
    table = mgr.dict()
    table.update([(b, v) for b, v in zip(boards_8ply, values_8ply)])
    all_boards = []
    all_values = []
    all_priors = []

    player = GridSearch("grid_det",
                        4,
                        ev.Evaluator(return_half))

    for i in [7, 6, 5, 4, 3, 2, 1, 0]:
        print(i)
        board_ips = list(make_random_ips(i))
        with Pool(processes=10) as pool:
            for boards, values, priors in pool.map(partial(backpropagate,
                                                           table=table,
                                                           player=player),
                                                   board_ips,
                                                   chunksize=int(1e4)):
                table.update((b, v) for b, v in zip(boards, values))
                all_boards.extend(boards)
                all_values.extend(values)
                all_priors.extend(priors)
                with open('/home/richard/Downloads/{}ply_boards.pth'.format(i), 'wb') as f:
                    pickle.dump(all_boards, f)
                with open('/home/richard/Downloads/{}ply_values.pth'.format(i), 'wb') as f:
                    pickle.dump(all_values, f)
                with open('/home/richard/Downloads/{}ply_priors.pth'.format(i), 'wb') as f:
                    pickle.dump(all_priors, f)

    print(all_boards[-1], all_values[-1], all_priors[-1])
    with open('/home/richard/Downloads/all_boards.pth', 'wb') as f:
        pickle.dump(all_boards, f)
    with open('/home/richard/Downloads/all_values.pth', 'wb') as f:
        pickle.dump(all_values, f)
    with open('/home/richard/Downloads/all_priors.pth', 'wb') as f:
        pickle.dump(all_priors, f)
    # torch.save(torch.stack(all_boards),
    #            open('~/Downloads/7ply_boards.pth'))
    # torch.save(torch.stack(all_values),
    #            open('~/Downloads/7ply_values.pth'))
    # torch.save(torch.stack(all_priors),
    #            open('~/Downloads/7ply_priors.pth'))

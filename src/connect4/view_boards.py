from src.connect4.board import Board

import numpy as np
import torch


def parse_line(line):
    x = line.split(',')
    if x[-1][0] == 'w':
        value = 1
    elif x[-1][0] == 'd':
        value = 0.5
    else:
        value = 0

    x = np.array([ord(i) for i in x[:-1]])
    x = np.flipud(np.transpose(x.reshape(7,6)))

    o_pieces = (x == 111)
    x_pieces = (x == 120)

    board = Board(o_pieces, x_pieces)
    return board, value


# f = open('/home/richard/Downloads/connect-4.data')

with open('/home/richard/Downloads/connect-4.data') as f:
    boards = []
    values = []
    both = []
    for line in f:
        board, value = parse_line(line)
        board = board.to_tensor()
        value = torch.tensor(float(value))
        boards.append(board)
        values.append(value)
        # both.append(torch.cat([board, value]))
    boards = torch.stack(boards)
    values = torch.stack(values)
    # both = torch.stack(both)

torch.save(boards, open('/home/richard/Downloads/connect4_boards.pth', 'wb'))
torch.save(values, open('/home/richard/Downloads/connect4_values.pth', 'wb'))
# torch.save(both, open('/home/richard/Downloads/connect4_both.pth', 'wb'))
# import pickle
# pickle.dump(posn, open('/home/richard/Downloads/connect4.pkl', 'wb'))

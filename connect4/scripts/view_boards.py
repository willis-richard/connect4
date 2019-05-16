from connect4.board import Board

import numpy as np
import torch


def parse_line(line):
    x = line.split(',')
    if x[-1][0] == 'w':
        value = 1.0
    elif x[-1][0] == 'd':
        value = 0.5
    else:
        value = 0.0

    x = np.array([ord(i) for i in x[:-1]])
    x = np.flipud(np.transpose(x.reshape(7,6)))

    o_pieces = (x == 111)
    x_pieces = (x == 120)

    board = Board(o_pieces, x_pieces)
    return board, value


# f = open('/home/richard/data/connect4/connect-4.data')

def read_8ply_data():
    with open('/home/richard/data/connect4/connect-4.data') as f:
        boards = []
        values = []
        for line in f:
            board, value = parse_line(line)
            boards.append(board)
            values.append(value)
    return boards, values


boards, values = read_8ply_data()

from oinkoink.utils import Connect4Stats as info
from oinkoink.utils import Result, Side

from copy import copy, deepcopy
import numpy as np
from typing import Set


WIDTH = info.width
HEIGHT = info.height
# my hash function converts to a u64
assert WIDTH * HEIGHT < 64
H1 = HEIGHT+1
H2 = HEIGHT+2
SIZE = HEIGHT*WIDTH
SIZE1 = H1*WIDTH
ALL1 = (1 << SIZE1)-1  # assumes SIZE1 < 63
COL1 = (1 << H1)-1
BOTTOM = int(ALL1 / COL1)  # has bits i*H1 set
TOP = BOTTOM << HEIGHT
BITMASK = np.flipud(np.transpose(np.reshape(
    np.array([2 ** x for x in range(SIZE1)]), ((H1,WIDTH)))))[1:,:]
# bitmask corresponds to board as follows in 7x6 case:
#  .  .  .  .  .  .  .  TOP
#  5 12 19 26 33 40 47
#  4 11 18 25 32 39 46
#  3 10 17 24 31 38 45
#  2  9 16 23 30 37 44
#  1  8 15 22 29 36 43
#  0  7 14 21 28 35 42  BOTTOM
HALF = int(WIDTH / 2)
SHIFT = (WIDTH - 1) * H1


class Board():
    def __init__(self):
        self.color = np.zeros((2,), dtype=np.int64)
        self.age = 0
        self.height = np.array([H1 * i for i in range(WIDTH)],
                               dtype=np.int64)
        self.result = None

    @classmethod
    def from_pieces(cls,
                    o_pieces,
                    x_pieces):
        board = cls()
        board.color[0] = np.dot(np.concatenate(o_pieces),
                                np.concatenate(BITMASK))
        board.color[1] = np.dot(np.concatenate(x_pieces),
                                np.concatenate(BITMASK))
        pieces = o_pieces + x_pieces
        board.age = np.sum(pieces)
        for i in range(WIDTH):
            board.height[i] = board.height[i] + np.count_nonzero(pieces[:, i])
        if board._check_terminal_position(board.color[0]):
            board.result = Result.o_win
        elif board._check_terminal_position(board.color[1]):
            board.result = Result.x_win
        elif board.age == SIZE:
            board.result = Result.draw
        return board

    @property
    def o_pieces(self):
        return np.flipud(np.reshape(
            np.array([(self.color[0] >> i) % 2
                      for i in range(SIZE1)],
                     dtype=np.bool_),
            (H1, WIDTH), order='F')[0:HEIGHT, :])

    @property
    def x_pieces(self):
        return np.flipud(np.reshape(
            np.array([(self.color[1] >> i) % 2
                      for i in range(SIZE1)],
                     dtype=np.bool_),
            (H1, WIDTH), order='F')[0:HEIGHT, :])

    @property
    def pieces(self):
        return self.o_pieces, self.x_pieces

    @property
    def player_to_move(self):
        return Side(self.age % 2)

    @property
    def valid_moves(self):
        if self.result is not None:
            return set()
        return set([i for i in range(WIDTH) if self._isplayable(i)])

    @property
    def symmetrical(self):
        return self.is_symmetrical(self.color[0]) and \
            self.is_symmetrical(self.color[1])

    def is_symmetrical(self, pieces):
        for i in range(HALF):
            # delete all columns to the right
            col_l = pieces << (SHIFT - H1 * i)
            col_l = col_l & ALL1
            # delete all columns to the left, leaving it in col 0
            col_l = col_l >> SHIFT
            # delete all columns to the right
            col_r = pieces << (H1 * i)
            col_r = col_r & ALL1
            # delete all columns to the left, leaving it in col 0
            col_r = col_r >> SHIFT
            if ((col_l ^ col_r) != 0):
                return False
        return True

    def create_fliplr(self):
        new_board = self.__class__()
        new_board.color[0] = self.flip_color(self.color[0])
        new_board.color[1] = self.flip_color(self.color[1])
        new_board.age = self.age
        base_height = np.array([H1 * i for i in range(WIDTH)],
                               dtype=np.int64)
        height_incr = self.height - base_height
        height_incr = np.flip(height_incr)
        new_board.height = base_height + height_incr
        new_board.result = deepcopy(self.result)
        return new_board

    def flip_color(self, pieces):
        new_pieces = 0
        for i in range(HALF):
            # take columns starting from the left
            col = pieces & (COL1 << (H1 * i))
            # shift it to where it should be (on rhs)
            col = col << (SHIFT - (2 * H1 * i))
            new_pieces += col
            # take columns starting from the right
            col = pieces & (COL1 << (SHIFT - H1 * i))
            # shift it to where it should be (on lhs)
            col = col >> (SHIFT - (2 * H1 * i))
            new_pieces += col
        if WIDTH % 2 == 1:
            # add back middle column
            col = pieces & (COL1 << (H1 * HALF))
            new_pieces += col
        return new_pieces

    def to_array(self):
        o_pieces, x_pieces = self.pieces

        to_move = np.ones(o_pieces.shape, dtype=np.uint8) if \
                      self.age % 2 == 0 else \
                      np.zeros(x_pieces.shape, dtype=np.uint8)

        return np.stack([to_move, o_pieces, x_pieces])

    def to_int_tuple(self):
        # FIXME: can I really have 1 value repr?
        return self.color[0], self.color[1]

    def make_move(self, move):
        self.color[self.age & 1] = \
            self.color[self.age & 1] ^ (1 << self.height[move])
        self.height[move] = self.height[move] + 1
        winner = self._check_terminal_position(self.color[self.age & 1])
        self.age += 1
        if winner:
            self.result = Result(self.age % 2)
        elif self.age == SIZE:
            self.result = Result.draw
        return self.result

    # return whether newboard includes a win
    def _check_terminal_position(self, newboard):
        y = newboard & (newboard >> HEIGHT)
        if ((y & (y >> 2*HEIGHT)) != 0):  # check diagonal \
            return True
        y = newboard & (newboard >> H1)
        if ((y & (y >> 2*H1)) != 0):  # check horizontal -
            return True
        y = newboard & (newboard >> H2)  # check diagonal /
        if ((y & (y >> 2*H2)) != 0):
            return True
        y = newboard & (newboard >> 1)  # check vertical |
        return (y & (y >> 2)) != 0

    # return whether columns col has room
    def _isplayable(self, col):
        return (self.color[self.age & 1] | (1 << self.height[col])) & TOP == 0

    def __copy__(self):
        new_board = self.__class__()
        new_board.color = deepcopy(self.color)
        new_board.age = self.age
        new_board.height = deepcopy(self.height)
        new_board.result = deepcopy(self.result)
        return new_board

    def __eq__(self, obj):
        return isinstance(obj, Board) \
            and np.array_equal(obj.color, self.color)

    def __hash__(self):
        return hash((self.color[0], self.color[1]))

    def __str__(self):
        o_pieces, x_pieces = self.pieces
        display = np.chararray(o_pieces.shape)
        display[:] = '-'
        display[o_pieces] = 'o'
        display[x_pieces] = 'x'
        return \
            str(np.array([range(WIDTH)]).astype(str)) \
            + "\n" + str(display.decode('utf-8')) \
            + "\n" + str(np.array([range(WIDTH)]).astype(str))

    def __repr__(self):
        return "color: {}, age: {}, height: {}, result: {}\n{}".format(
            self.color,
            self.age,
            self.height,
            self.result,
            self.__str__())


def make_random_ips(plies):
    ips = set()
    board = Board()
    expand(ips, board, plies)
    return ips


def expand(ips: Set,
           board: Board,
           plies: int) -> None:
    if plies == 0:
        if board.result is None:
            ips.add(board)
        return
    valid_moves = board.valid_moves
    for move in valid_moves:
        new_board = copy(board)
        new_board.make_move(move)
        expand(ips, new_board, plies - 1)

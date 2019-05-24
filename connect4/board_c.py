import numpy as np


WIDTH = 7
HEIGHT = 6
# bitmask corresponds to board as follows in 7x6 case:
#  .  .  .  .  .  .  .  TOP
#  5 12 19 26 33 40 47
#  4 11 18 25 32 39 46
#  3 10 17 24 31 38 45
#  2  9 16 23 30 37 44
#  1  8 15 22 29 36 43
#  0  7 14 21 28 35 42  BOTTOM
H1 = HEIGHT+1
H2 = HEIGHT+2
SIZE = HEIGHT*WIDTH
SIZE1 = H1*WIDTH
ALL1 = (np.uint64(1) << SIZE1)-np.uint64(1)  # assumes SIZE1 < 63
COL1 = (np.uint64(1) << H1)-np.uint64(1)
BOTTOM = ALL1 / COL1  # has bits i*H1 set
TOP = BOTTOM << HEIGHT


class Board_c():
    def __init__(self):
        self.color = np.zeros((2,), dtype=np.uint64)
        self.nplies = np.uint8(0)
        self.height = np.zeros([H1 * i for i in range(WIDTH)],
                               dtype=np.byte)

    @property
    def positioncode(self):
        # color[0] + color[1] + BOTTOM forms bitmap of heights
        # so that positioncode() is a complete board encoding
        return 2*self.color[0] + self.color[1] + BOTTOM

    # return whether columns col has room
    def isplayable(self, col):
        newboard = np.uint64(self.color[self.nplies & 1] |
                             (np.uint64(1) << self.height[col]))
        return self.islegal(newboard)

    # return whether newboard lacks overflowing column
    def islegal(self, newboard):
        return (newboard & TOP) == 0

    # return whether newboard is legal and includes a win
    def islegalhaswon(self, newboard):
        return self.islegal(newboard) and self.haswon(newboard)

    # return whether newboard includes a win
    def haswon(self, newboard):
        y = np.uint64(newboard & (newboard >> HEIGHT))
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

    def make_move(self, n):
        self.color[self.nplies & 1] = self.color[self.nplies & 1] ^ \
                                      (np.uint64(1) << (self.height[n] + 1))
        self.nplies += 1

    def __eq__(self, obj):
        return isinstance(obj, Board_c) \
            and np.array_equal(obj.colors, self.colors)

    def __hash__(self):
        return hash(self.colors)

    @property
    def pieces(self):
        o_pieces = np.array([(self.color[0] >> i) % 2
                             for i in range(SIZE1)],
                            dtype=np.uint8)
        o_pieces = np.reshape((H1, WIDTH), o_pieces)[0:HEIGHT, :]
        x_pieces = np.array([(self.color[1] >> i) % 2
                             for i in range(SIZE1)],
                            dtype=np.uint8)
        x_pieces = np.reshape((H1, WIDTH), x_pieces)[0:HEIGHT, :]
        return o_pieces, x_pieces

    def to_array(self):
        to_move = np.ones(self.o_pieces.shape, dtype=np.uint8) if \
                  self.nplies % 2 == 0 else \
                  np.zeros(self.x_pieces.shape, dtype=np.uint8)

        o_pieces, x_pieces = self.pieces

        return np.stack([to_move, o_pieces, x_pieces])

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

    @property
    def valid_moves(self):
        return set([self.isplayable(i) for i in range(WIDTH)])


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

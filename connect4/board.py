from connect4.utils import Connect4Stats as info
from connect4.utils import Side, Result

from copy import copy
import numpy as np
from typing import Set


class Board():
    def __init__(self,
                 o_pieces=None,
                 x_pieces=None):
        self.o_pieces = np.zeros((info.height, info.width), dtype=np.bool_) if\
            o_pieces is None else o_pieces
        self.x_pieces = np.zeros((info.height, info.width), dtype=np.bool_) if\
            x_pieces is None else x_pieces
        self.result = None

        if o_pieces is None and x_pieces is None:
            self._player_to_move = Side.o
        else:
            self.player_to_move = Side(1 - (np.count_nonzero(self.o_pieces) - np.count_nonzero(self.x_pieces)))
            self._check_valid()
            self.check_terminal_position()

    def __copy__(self):
        new_board = self.__class__()
        new_board.o_pieces = copy(self.o_pieces)
        new_board.x_pieces = copy(self.x_pieces)
        new_board._player_to_move = copy(self._player_to_move)
        new_board.result = copy(self.result)
        return new_board

    def __eq__(self, obj):
        return isinstance(obj, Board) \
            and np.array_equal(obj.o_pieces, self.o_pieces) \
            and np.array_equal(obj.x_pieces, self.x_pieces)

    @property
    def player_to_move(self):
        return 'o' if self._player_to_move == Side.o else 'x'

    @player_to_move.setter
    def player_to_move(self, player_to_move: Side):
        self._player_to_move = player_to_move

    @property
    def age(self):
        return np.sum(self._get_pieces())

    def __repr__(self):
        display = np.chararray(self.o_pieces.shape)
        display[:] = '-'
        display[self.o_pieces] = 'o'
        display[self.x_pieces] = 'x'
        return \
            str(np.array([range(info.width)]).astype(str)) \
            + "\n" + str(display.decode('utf-8')) \
            + "\n" + str(np.array([range(info.width)]).astype(str))

    def to_array(self):
        to_move = np.ones(self.o_pieces.shape, dtype=np.bool_) if \
                  self._player_to_move == Side.o else \
                  np.zeros(self.x_pieces.shape, dtype=np.bool_)

        return np.stack([to_move.astype(np.uint8),
                         self.o_pieces.astype(np.uint8),
                         self.x_pieces.astype(np.uint8)])

    def _get_pieces(self):
        return self.o_pieces + self.x_pieces

    @property
    def valid_moves(self):
        if self.result is not None:
            return set()
        pieces = self._get_pieces()
        return set(i for i in range(info.width) if not all(pieces[:, i]))

    def get_plies(self):
        return np.sum(self._get_pieces())

    def _check_straight(self, pieces):
        return np.any(np.all([pieces[i:i+info.win_length, j] for i in range(pieces.shape[0] - info.win_length + 1) for j in range(pieces.shape[1])], axis=1))

    def _check_diagonal(self, pieces):
        return np.any(np.all(np.diagonal([pieces[i:i+info.win_length, j:j+info.win_length] for i in range(pieces.shape[0] - info.win_length + 1) for j in range(pieces.shape[1] - info.win_length + 1)], axis1=1, axis2=2), axis=1))

    def check_for_winner(self, pieces):
        return \
            self._check_straight(pieces) or \
            self._check_straight(np.transpose(pieces)) or \
            self._check_diagonal(pieces) or \
            self._check_diagonal(np.fliplr(pieces))

    def check_terminal_position(self, check_all: bool = True):
        if (check_all or self._player_to_move == Side.x) and \
           self.check_for_winner(self.o_pieces):
            self.result = Result.o_win
        elif (check_all or self._player_to_move == Side.o) and \
             self.check_for_winner(self.x_pieces):
            self.result = Result.x_win
        elif np.all(self._get_pieces()):
            self.result = Result.draw
        return self.result

    def _make_move(self, move):
        board = self._get_pieces()
        idx = info.height - np.count_nonzero(board[:, move]) - 1
        if self._player_to_move == Side.o:
            self.o_pieces[idx, move] = 1
        else:
            self.x_pieces[idx, move] = 1
        self.player_to_move = Side(1 - self._player_to_move.value)

    def make_move(self, move):
        self._make_move(move)
        return self.check_terminal_position()

    def _check_valid(self):
        assert self.o_pieces.shape == (info.height, info.width)
        assert self.x_pieces.shape == (info.height, info.width)

        no_gaps = True
        for col in range(info.width):
            board = self._get_pieces()
            if not np.array_equal(board[:, col], np.sort(board[:, col])):
                no_gaps = False

        assert no_gaps
        assert not np.any(np.logical_and(self.o_pieces, self.x_pieces))
        assert np.sum(self.o_pieces) - np.sum(self.x_pieces) == (1 - self._player_to_move.value)
        # check the player to move has not already won
        assert not (self.check_for_winner(self.o_pieces) and self._player_to_move == Side.o)
        assert not (self.check_for_winner(self.x_pieces) and self._player_to_move == Side.x)

    def __hash__(self):
        o_hash = np.dot(np.concatenate(self.o_pieces), info.hash_value)
        x_hash = np.dot(np.concatenate(self.x_pieces), info.hash_value)
        return hash((o_hash, x_hash))


def make_random_ips(plies):
    ips = set()
    board = Board()
    expand(ips, board, plies)
    return ips


def expand(ips: Set,
           board: Board,
           plies: int) -> None:
    if plies == 0:
        ips.add(board)
        return
    for move in board.valid_moves:
        new_board = copy(board)
        new_board.make_move(move)
        expand(ips, new_board, plies - 1)

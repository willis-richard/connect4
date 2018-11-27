from connect4.utils import advance_player

import numpy as np


class Board():
    def __init__(self,
                 hash_value,
                 height=6,
                 width=7,
                 win_length=4,
                 o_pieces=None,
                 x_pieces=None):
        self._width = width
        self._height = height
        self._win_length = win_length
        self.o_pieces = np.zeros((self._height, self._width), dtype=np.bool_) if o_pieces is None else o_pieces
        self.x_pieces = np.zeros((self._height, self._width), dtype=np.bool_) if x_pieces is None else x_pieces

        self.player_to_move = np.count_nonzero(self.o_pieces) - np.count_nonzero(self.x_pieces)
        self.move_history = []
        self.result = None

        self._check_valid()

    def __eq__(self, obj):
        return isinstance(obj, Board) \
            and obj.o_pieces == self.o_pieces \
            and obj.x_pieces == self.x_pieces

    @property
    def player_to_move(self):
        return 'o' if self._player_to_move == 0 else 'x'

    @player_to_move.setter
    def player_to_move(self, player_to_move):
        assert player_to_move in [0, 1]
        self._player_to_move = player_to_move

    def display(self):
        display = np.chararray(self.o_pieces.shape)
        display[:] = ' '
        display[self.o_pieces] = 'o'
        display[self.x_pieces] = 'x'
        print(display.decode('utf-8'))

    def _get_pieces(self):
        return self.o_pieces + self.x_pieces

    def valid_moves(self):
        pieces = self._get_pieces()
        return set(i for i in range(self._width) if not all(pieces[:,i]))

    def get_half_moves(self):
        return np.sum(self._get_pieces())

    def _check_straight(self, pieces):
        """Returns the number of horizontal wins"""
        count = 0
        for i in range(pieces.shape[1]):
            for j in range(pieces.shape[0] - self._win_length + 1):
                if np.all(pieces[j:j+self._win_length, i]):
                    count += 1
        return count

    def _check_diagonal(self, pieces):
        """Returns the number of diagonally down and right winners"""
        count = 0
        for i in range(pieces.shape[0] - self._win_length + 1):
            for j in range(pieces.shape[1] - self._win_length + 1):
                if np.count_nonzero([pieces[i+x, j+x] for x in range(self._win_length)]) == self._win_length:
                    count += 1
        return count

    def check_for_winner(self, pieces=None):
        if pieces is None:
            pieces = self.x_pieces if self._player_to_move else self.o_pieces

        return \
            self._check_straight(pieces) + \
            self._check_straight(np.transpose(pieces)) + \
            self._check_diagonal(pieces) + \
            self._check_diagonal(np.fliplr(pieces))

    def check_terminal_position(self):
        if self.check_for_winner(self.o_pieces):
            self.result = 1
        elif self.check_for_winner(self.x_pieces):
            self.result = -1
        elif np.all(self._get_pieces()):
            self.result = 1

    def make_move(self, move):
        board = self._get_pieces()
        idx = self._height - np.count_nonzero(board[:, move]) - 1
        if self._player_to_move:
            self.x_pieces[idx, move] = 1
        else:
            self.o_pieces[idx, move] = 1
        self.player_to_move = advance_player(self._player_to_move)
        # FIXME: remove when working
        self._check_valid()
        self.check_terminal_position()

    def _check_valid(self):
        no_gaps = True
        for col in range(self._width):
            board = self._get_pieces()
            if not np.array_equal(board[:, col], np.sort(board[:, col])):
                no_gaps = False

        assert no_gaps
        assert not np.any(np.logical_and(self.o_pieces, self.x_pieces))
        assert np.sum(self.o_pieces) - np.sum(self.x_pieces)  in [0, 1]
        #assert self.check_for_winner(self.o_pieces) + self.check_for_winner(self.x_pieces) in [0,1]
        assert (self.check_for_winner(self.o_pieces) == 0) or (self._player_to_move == 1)
        assert (self.check_for_winner(self.x_pieces) == 0) or (self._player_to_move == 0)
        assert self.o_pieces.shape == (self._height, self._width)
        assert self.x_pieces.shape == (self._height, self._width)

    def __hash__(self):
        o_hash = np.dot(np.concatenate(self.o_pieces), self.hash_value)
        x_hash = np.dot(np.concatenate(self.x_pieces), self.hash_value)
        return hash((o_hash, x_hash))

import numpy as np


class Board():
    def __init__(self,
                 height=6,
                 width=7,
                 win_length=4,
                 white_pieces=None,
                 black_pieces=None):
        # Check inputs are sane
        assert width >= win_length and height >= win_length

        self._width = width
        self._height = height
        self._win_length = win_length
        self.white_pieces = np.zeros((self._height, self._width), dtype=np.bool_) if white_pieces is None else white_pieces
        self.black_pieces = np.zeros((self._height, self._width), dtype=np.bool_) if black_pieces is None else black_pieces
        print(self.white_pieces)
        print(self.black_pieces)
        print("111111111111111111111111111", np.count_nonzero(self.white_pieces) - np.count_nonzero(self.black_pieces))

        self.player_to_move = np.count_nonzero(self.white_pieces) - np.count_nonzero(self.black_pieces)

        # Check members are sane
        assert self.white_pieces.shape == (self._height, self._width)
        assert self.black_pieces.shape == (self._height, self._width)

    @property
    def player_to_move(self):
        return self.__player_to_move

    @player_to_move.setter
    def player_to_move(self, player_to_move):
        assert player_to_move in [0, 1]
        self.__player_to_move = player_to_move

    @player_to_move.getter
    def player_to_move(self):
        return 'o' if self.__player_to_move == 0 else 'x'

    def display(self):
        display = np.chararray(self.white_pieces.shape)
        display[:] = ' '
        display[self.white_pieces] = 'o'
        display[self.black_pieces] = 'x'
        print(display.decode('utf-8'))

    def _advance_player(self):
        self.player_to_move = (self.__player_to_move + 1) % 2

    def _get_pieces(self):
        return self.white_pieces + self.black_pieces

    def get_valid_moves(self):
        pieces = self._get_pieces()
        return [i for i in range(self._width) if not all(pieces[:,i])]

    def _check_straight(self, pieces):
        """Checks for horizontal wins"""
        for i in range(pieces.shape[1]):
            for j in range(pieces.shape[0] - self._win_length + 1):
                if np.all(pieces[j:j+self._win_length, i]):
                    return True
        return False

    def _check_diagonal(self, pieces):
        for i in range(pieces.shape[1] - self._win_length):
            for j in range(pieces.shape[0] - self._win_length):
                if np.count_nonzero([pieces[i+x,j+x] for x in range(self._win_length)]) == self._win_length:
                    return True
        return False

    def check_for_winner(self, pieces=None):
        if pieces is None:
            pieces = self.black_pieces if self.__player_to_move else self.white_pieces

        return \
            self._check_straight(pieces) or \
            self._check_straight(np.transpose(pieces)) or \
            self._check_diagonal(pieces) or \
            self._check_diagonal(np.flipud(pieces))

    def check_terminal_position(self):
        if self.check_for_winner(self.white_pieces):
            return 1
        elif self.check_for_winner(self.black_pieces):
            return 2
        elif np.all(self._get_pieces()):
            return 3
        return 0

    def make_move(self, move):
        board = self._get_pieces()
        idx = self._height - np.count_nonzero(board[:, move]) - 1
        if self.__player_to_move:
            self.black_pieces[idx, move] = 1
        else:
            self.white_pieces[idx, move] = 1
        # FIXME: remove when working
        assert self.check_valid()
        self._advance_player()

    def check_valid(self):
        no_gaps = True
        for col in range(self._width):
            board = self._get_pieces()
            if not np.array_equal(board[:, col], np.sort(board[:, col])):
                no_gaps = False
        return \
            no_gaps and \
            not np.any(np.logical_and(self.white_pieces, self.black_pieces)) and \
            np.sum(self.white_pieces) - np.sum(self.black_pieces) == self.__player_to_move


class Player():
    def __init__(self):
        return

    def make_move(self):
        return


if __name__ == "__main__":
    board = Board()
    board.display()
    while not board.check_terminal_position():
        print("FYI valid moves are: ", board.get_valid_moves())
        move = int(input("Enter player " + board.player_to_move + "'s move:"))
        while move not in board.get_valid_moves():
            print("Try again dipshit")
            move = int(input("Enter player " + board.player_to_move + "'s move:"))
        board.make_move(move)
        board.display()

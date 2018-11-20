import copy

import numpy as np

# Utility functions
def advance_player(player_to_move):
    return (player_to_move + 1) % 2

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

        self.player_to_move = np.count_nonzero(self.white_pieces) - np.count_nonzero(self.black_pieces)
        self.result = None

        self._check_valid()

    @property
    def player_to_move(self):
        return 'o' if self._player_to_move == 0 else 'x'

    @player_to_move.setter
    def player_to_move(self, player_to_move):
        assert player_to_move in [0, 1]
        self._player_to_move = player_to_move

    def display(self):
        display = np.chararray(self.white_pieces.shape)
        display[:] = ' '
        display[self.white_pieces] = 'o'
        display[self.black_pieces] = 'x'
        print(display.decode('utf-8'))

    def _get_pieces(self):
        return self.white_pieces + self.black_pieces

    def valid_moves(self):
        pieces = self._get_pieces()
        return [i for i in range(self._width) if not all(pieces[:,i])]

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
            pieces = self.black_pieces if self._player_to_move else self.white_pieces

        return \
            self._check_straight(pieces) + \
            self._check_straight(np.transpose(pieces)) + \
            self._check_diagonal(pieces) + \
            self._check_diagonal(np.fliplr(pieces))

    def check_terminal_position(self):
        if self.check_for_winner(self.white_pieces):
            self.result = 0
        elif self.check_for_winner(self.black_pieces):
            self.result = 1
        elif np.all(self._get_pieces()):
            self.result = 2

    def make_move(self, move):
        board = self._get_pieces()
        idx = self._height - np.count_nonzero(board[:, move]) - 1
        if self._player_to_move:
            self.black_pieces[idx, move] = 1
        else:
            self.white_pieces[idx, move] = 1
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
        assert not np.any(np.logical_and(self.white_pieces, self.black_pieces))
        assert np.sum(self.white_pieces) - np.sum(self.black_pieces)  in [0, 1]
        assert self.check_for_winner(self.white_pieces) + self.check_for_winner(self.black_pieces) in [0,1]
        assert (self.check_for_winner(self.white_pieces) == 0) or (self._player_to_move == 1)
        assert (self.check_for_winner(self.black_pieces) == 0) or (self._player_to_move == 0)
        assert self.white_pieces.shape == (self._height, self._width)
        assert self.black_pieces.shape == (self._height, self._width)


class BasePlayer():
    def __init__(self, name, side, board):
        self.name = name
        self.side = side
        self.enemy_side = advance_player(side)
        self._board = board

    def __str__(self):
        return "Player: " + self.name


class HumanPlayer(BasePlayer):
    def __init__(self, name, side, board):
        super().__init__(name, side, board)

    def make_move(self):
        print("FYI valid moves are: ", self._board.valid_moves())
        move = str(input("Enter player " + self._board.player_to_move + "'s move:"))
        while len(move) != 1 or int(move) not in self._board.valid_moves():
            print("Try again dipshit")
            move = str(input("Enter player " + self._board.player_to_move + "'s move:"))
        self._board.make_move(int(move))

    def __str__(self):
        return super().__str__() + ", type: Human"



class ComputerPlayer(BasePlayer):
    def __init__(self, name, side, board):
        super().__init__(name, side, board)

    def make_move(self):
        moves = self._board.valid_moves()

        for move in moves:
            new_board = copy.deepcopy(self._board)
            new_board.make_move(move)
            if new_board.result == self.side:
                print("Trash! I will crush you.")
                self._board.make_move(move)
                return

        moves = np.array(moves)
        distance_to_middle = np.abs(moves - self._board._width / 2.0)
        idx = np.argsort(distance_to_middle)
        moves = moves[idx]

        to_print = False
        for move in moves:
            new_board = copy.deepcopy(self._board)
            new_board.make_move(move)
            enemy_moves = new_board.valid_moves()
            safe = True
            for enemy_move in enemy_moves:
                new_enemy_board = copy.deepcopy(new_board)
                new_enemy_board.make_move(enemy_move)
                if new_enemy_board.result == self.enemy_side:
                    safe = False
                    to_print = True
                    break
            if safe:
                if to_print:
                    print("zzz as if")
                self._board.make_move(move)
                return

        print("Ah fuck you lucky shit")
        self._board.make_move(np.random.choice(moves))

    def __str__(self):
        return super().__str__() + ", type: Computer"


class Connect4():
    def __init__(self, player_o_name, player_x_name, player_o_human=False, player_x_human=False):
        self._board = Board()
        self._player_o = HumanPlayer(player_o_name, 0, self._board) if player_o_human else ComputerPlayer(player_o_name, 0, self._board)
        self._player_x = HumanPlayer(player_x_name, 1, self._board) if player_x_human else ComputerPlayer(player_x_name, 1, self._board)

    def play(self):
        print("Match between", self._player_o, " and ", self._player_x)
        self._board.display()
        while not self._board.result:
            if self._board.player_to_move == 'o':
                self._player_o.make_move()
            else:
                self._player_x.make_move()
            self._board.display()

        if self._board.result == 1:
            result = "o wins"
        elif self._board.result == 2:
            result = "x wins"
        else:
            result = "draw"

        print("The result is: ", result)


if __name__ == "__main__":
    game = Connect4("A", "B", True, False)
    game.play()

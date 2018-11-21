import copy

from connect4.utils import advance_player

import numpy as np


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

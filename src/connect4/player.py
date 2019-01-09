from src.connect4 import tree
from src.connect4.utils import Side


class BasePlayer():
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "Player: " + self.name

    @property
    def side(self) -> Side:
        return self._side

    @side.setter
    def side(self, side: Side):
        self._side = side


class HumanPlayer(BasePlayer):
    def __init__(self, name):
        super().__init__(name)

    def make_move(self, board):
        move = -1
        while move not in board.valid_moves:
            try:
                move = int(input("Enter " + self.name + " (" +
                                 board.player_to_move + "'s) move:"))
            except ValueError:
                print("Try again dipshit")
                pass
        board.make_move(int(move))
        return move

    def __str__(self):
        return super().__str__() + ", type: Human"


class ComputerPlayer(BasePlayer):
    def __init__(self, name, strategy):
        super().__init__(name)
        self.tree = tree.Tree(strategy.Evaluation)
        self.search_fn = strategy.get_search_fn()

    def make_move(self, board):
        self.tree.update_root(board)
        move, value = self.search_fn(tree=self.tree,
                                     board=board,
                                     side=self.side)

        if value == self.side:
            print("Trash! I will crush you.")
        elif value == -1 * self.side:
            print("Ah fuck you lucky shit")

        print(self.name + " selected move: ", move)
        board.make_move(move)

        return move

    def __str__(self):
        return super().__str__() + ", type: Computer"

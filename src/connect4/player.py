from src.connect4 import tree


class BasePlayer():
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "Player: " + self.name


class HumanPlayer(BasePlayer):
    def __init__(self, name):
        super().__init__(name)

    def make_move(self, board):
        super().make_move(board)
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
        self.tree = tree.Tree(strategy.NodeData,
                              strategy.PositionEvaluation)
        self.search_fn = strategy.get_search_fn()

    def get_move(self, board):
        side = board._player_to_move
        self.tree.update_root(board)
        move, value = self.search_fn(tree=self.tree,
                                     board=board,
                                     side=side)

        if value == 1.0:
            print("Trash! I will crush you.")
        elif value == 0.0:
            print("Ah fuck you lucky shit")

        print(self.name + " selected move: ", move)

        return move

    def make_move(self, board):
        move = self.get_move(board)
        board.make_move(move)
        return move

    def __str__(self):
        return super().__str__() + ", type: Computer"

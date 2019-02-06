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
    def __init__(self, name, strategy, nn_storage=None):
        super().__init__(name)
        self.tree = tree.Tree(strategy.NodeData,
                              strategy.PositionEvaluation)
        if nn_storage is None:
            self.search_fn = strategy.get_search_fn()
        else:
            net = nn_storage.get_net()
            self.search_fn = strategy.get_search_fn(net)

    def get_move(self, board):
        side = board._player_to_move
        self.tree.update_root(board)
        move, value, policy = self.search_fn(tree=self.tree,
                                             board=board,
                                             side=side)

        return move, value, policy

    def make_move(self, board):
        move, value, policy = self.get_move(board)
        board.make_move(move)
        return move, value, policy

    def __str__(self):
        return super().__str__() + ", type: Computer"

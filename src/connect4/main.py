"""Hi"""

from src.connect4 import match
from src.connect4 import player
from src.connect4 import searching

if __name__ == "__main__":
    # player_1 = player.HumanPlayer("player_1")
    player_1 = player.ComputerPlayer("player_1",
                                     searching.GridSearch(plies=4))
    # player_2 = player.ComputerPlayer("player_2",
    #                                  searching.GridSearch(plies=4))
    player_2 = player.ComputerPlayer("player_2",
                                     searching.MCTS(simulations=30))
    # match = match.Match(True, 24, player_1, player_2)
    # match.play(agents=12)
    match = match.Match(True, 1, player_1, player_2)
    match.play(agents=1)

"""Hi"""

from src.connect4.match import Match
from src.connect4 import player
from src.connect4.searching import GridSearch, MCTS

from src.connect4.neural.config import AlphaZeroConfig
from src.connect4.neural.training import TrainingLoop

import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # player_1 = player.HumanPlayer("human_1")
        player_1 = player.ComputerPlayer("grid_1",
                                         GridSearch(plies=4))
        # player_1 = player.ComputerPlayer("mcts_1",
        #                                  MCTS(MCTS.Config(simulations=2500,
        #                                                   cpuct=9999)))
        player_2 = player.ComputerPlayer("grid_2",
                                         GridSearch(plies=4))
        # player_2 = player.ComputerPlayer("mcts_2",
        #                                  MCTS(MCTS.Config(simulations=2500,
        #                                                   cpuct=9999)))
        match = Match(True, 24, player_1, player_2)
        match.play(agents=12)
        # match = Match(True, 1, player_1, player_2)
        # match.play(agents=1)
    else:
        folder_path = '/home/richard/Downloads/nn/new_dir'
        TrainingLoop(AlphaZeroConfig,
                     folder_path).run()

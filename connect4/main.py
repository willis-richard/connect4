"""Hi"""

import connect4.evaluators as evaluators
from connect4.grid_search import GridSearch
from connect4.match import Match
from connect4.mcts import MCTS, MCTSConfig
from connect4.player import HumanPlayer

from connect4.neural.config import AlphaZeroConfig
from connect4.neural.training import TrainingLoop

import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # player_1 = HumanPlayer("human_1")
        player_1 = GridSearch("grid_1",
                              4,
                              evaluators.Evaluator(
                                  evaluators.evaluate_centre))
        # player_1 = player.ComputerPlayer("mcts_1",
        #                                  MCTS(MCTS.Config(simulations=2500,
        #                                                   pb_c_init=9999)))
        # player_2 = player.ComputerPlayer("grid_2",
        #                                  GridSearch(plies=4))
        player_2 = MCTS("mcts_2",
                        MCTSConfig(simulations=2500,
                                   pb_c_init=9999),
                        evaluators.Evaluator(
                            evaluators.evaluate_centre_with_prior))
        match = Match(True, player_1, player_2, plies=1, switch=True)
        # match.play(agents=12)
        # match = Match(True, player_1, player_2, plies=0, switch=False)
        match.play(agents=2)
    else:
        TrainingLoop(AlphaZeroConfig).run()

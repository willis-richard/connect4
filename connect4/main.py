"""Hi"""

import connect4.evaluators as ev
from connect4.grid_search import GridSearch
from connect4.match import Match
from connect4.mcts import MCTS, MCTSConfig
from connect4.player import HumanPlayer

from connect4.neural.config import AlphaZeroConfig, ModelConfig
from connect4.neural.nn_pytorch import ModelWrapper
from connect4.neural.training import TrainingLoop

import argparse
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
import sys
from torch.multiprocessing import set_start_method


class Parser():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Run either a single game, or use AlphaZero either in a match or training loop.',
                                         usage='<game|match|training> [<args>]')
        parser.add_argument('-m', '--mode',
                            choices=['game', 'match', 'training'],
                            help='Choose whether to run a match or a training loop')
        self.mode = parser.parse_args(sys.argv[1:3])

        if not hasattr(self, self.mode.mode):
           print('Unrecognized command')
           parser.print_help()
           exit(1)

        getattr(self, self.mode.mode)()

    def game(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        parser = argparse.ArgumentParser(description='Run a game between two human players.')
        parser.add_argument('-p', '--players', nargs=2,
                            help='the names of the two players')
        parser.add_argument('-n', '--net_filepath', type=str, required=False,
                            default=dirname + '/data/example_net.pth',
                            help='filepath to a pytorch network')
        self.args = parser.parse_args(sys.argv[3:])

    def match(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        parser = argparse.ArgumentParser(description='Run a match from different initial positions.')
        parser.add_argument('-a', '--agents', default=1, type=int,
                            help='how many processes to run')
        parser.add_argument('-p', '--plies', default=1, type=int,
                            help='the number of pre-made half-moves for each game')
        parser.add_argument('-s', '--switch', default=False, type=bool,
                            help='play again but reverse first player for each position?')
        parser.add_argument('-d', '--display', default=False, type=bool,
                            help='display each position?')
        parser.add_argument('-n', '--net_filepath', type=str, required=False,
                            default=dirname + '/data/example_net.pth',
                            help='filepath to a pytorch network')
        self.args = parser.parse_args(sys.argv[3:])

    def training(self):
        parser = argparse.ArgumentParser(description='Run a training loop.')
        parser.add_argument('-c', '--config', required=False,
                            help='An AlphaZero config filepath')
        self.args = parser.parse_args(sys.argv[3:])

def main():
    try:
        set_start_method('spawn')
    except RuntimeError as e:
        if str(e) == 'context has already been set':
            pass

    parser = Parser()
    if parser.mode.mode == 'game':
        player_1 = HumanPlayer(parser.args.players[0])
        # player_2 = HumanPlayer(parser.args.players[1])
        model = ModelWrapper(ModelConfig(),
                             file_name=parser.args.net_filepath)

        player_2 = MCTS(parser.args.players[0],
                        MCTSConfig(simulations=800),
                        ev.Evaluator(partial(ev.evaluate_nn,
                                             model=model)))

        match = Match(True, player_1, player_2, switch=True)
        match.play()
    elif parser.mode.mode == 'match':
        player_1 = GridSearch("grid_det",
                              4,
                              ev.Evaluator(ev.evaluate_centre))

        player_2 = MCTS("mcts_det",
                        MCTSConfig(simulations=2000),
                        ev.Evaluator(ev.evaluate_centre_with_prior))

        # model = ModelWrapper(ModelConfig(),
        #                      file_name=parser.args.net_filepath)

        # player_3 = MCTS("mcts_nn",
        #                 MCTSConfig(simulations=2000),
        #                 ev.Evaluator(partial(ev.evaluate_nn,
        #                                      model=model)))

        match = Match(parser.args.display,
                      player_2,
                      player_2,
                      plies=parser.args.plies,
                      switch=parser.args.switch)
        match.play(agents=parser.args.agents)
    else:
        if parser.args.config:
            spec = spec_from_file_location("module.name", parser.args.config)
            config = module_from_spec(spec)
            spec.loader.exec_module(config)
            config = config.config
        else:
            config = AlphaZeroConfig()

        TrainingLoop(config).run()

if __name__ == "__main__":
    main()

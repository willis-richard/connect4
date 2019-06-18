import connect4.evaluators as ev
from connect4.match import Match
from connect4.mcts import MCTS, MCTSConfig
from connect4.player import HumanPlayer

from connect4.neural.config import AlphaZeroConfig, ModelConfig
from connect4.neural.pytorch.model import ModelWrapper
from connect4.neural.training import TrainingLoop

import argparse
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
import os
import sys


class Parser():
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Either play a game vs the AI or run the training loop',
            usage='<game|training> [<args>]')
        parser.add_argument('-m', '--mode',
                            choices=['game', 'training'],
                            help='What mode to run')
        self.mode = parser.parse_args(sys.argv[1:3])

        if self.mode.mode is None:
            print('Please choose a command')
            parser.print_help()
            exit(1)

        getattr(self, self.mode.mode)()

    def game(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        parser = argparse.ArgumentParser(
            description='Play against the agent')
        parser.add_argument('-n', '--net_filepath', type=str, required=False,
                            default=dirname + '/data/example_net.pth',
                            help='filepath to a pytorch network')
        self.args = parser.parse_args(sys.argv[3:])

    def training(self):
        parser = argparse.ArgumentParser(description='Run a training loop')
        parser.add_argument('-c', '--config', required=False,
                            help='An AlphaZero config filepath')
        self.args = parser.parse_args(sys.argv[3:])


def main():
    parser = Parser()
    if parser.mode.mode == 'game':
        player_1 = HumanPlayer('User')
        model = ModelWrapper(ModelConfig(),
                             file_name=parser.args.net_filepath)

        player_2 = MCTS('AI',
                        MCTSConfig(simulations=800),
                        ev.Evaluator(partial(ev.evaluate_nn,
                                             model=model)))

        match = Match(True, player_1, player_2, switch=True)
        match.play()
    else:
        from torch.multiprocessing import set_start_method
        try:
            set_start_method('spawn')
        except RuntimeError as e:
            if str(e) == 'context has already been set':
                pass

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

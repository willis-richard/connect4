import oinkoink.evaluators as ev
from oinkoink.match import Match
from oinkoink.mcts import MCTS, MCTSConfig
from oinkoink.player import HumanPlayer

from oinkoink.neural.config import AlphaZeroConfig, ModelConfig
from oinkoink.neural.pytorch.model import ModelWrapper
from oinkoink.neural.training import TrainingLoop

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
        parser.add_argument('-s', '--simulations', type=int, required=False,
                            default=800,
                            help='Number of positions the AI will evaluate each move.'
                            'Minimum value of 1')
        self.args = parser.parse_args(sys.argv[3:])
        if not os.path.isfile(self.args.net_filepath):
            raise FileNotFoundError('net_filepath incorrectly specified')
        if self.args.simulations <= 0:
            raise ValueError('Simlulations must be a positive integer')

    def training(self):
        parser = argparse.ArgumentParser(description='Run a training loop')
        parser.add_argument('-c', '--config', required=True,
                            help='An AlphaZero config filepath')
        self.args = parser.parse_args(sys.argv[3:])


def main():
    parser = Parser()
    if parser.mode.mode == 'game':
        player_1 = HumanPlayer('User')
        model = ModelWrapper(ModelConfig(use_gpu=False),
                             file_name=parser.args.net_filepath)

        player_2 = MCTS('AI',
                        MCTSConfig(simulations=parser.args.simulations),
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

        spec = spec_from_file_location('module.name', parser.args.config)
        config = module_from_spec(spec)
        spec.loader.exec_module(config)
        config = config.config

        TrainingLoop(config).run()


if __name__ == '__main__':
    main()

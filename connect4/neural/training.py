from connect4.board_c import Board
import connect4.evaluators as evl
from connect4.grid_search import GridSearch
from connect4.match import Match
from connect4.mcts import MCTS, MCTSConfig
from connect4.utils import Result

from connect4.neural.config import AlphaZeroConfig
from connect4.neural.game_pool import game_pool
from connect4.neural.inference_server import InferenceServer
from connect4.neural.training_game import training_game

from functools import partial
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from torch.multiprocessing import Pipe, Pool
from visdom import Visdom

if True:
    from connect4.neural.nn_pytorch import (Connect4Dataset,
                                            ModelWrapper,
                                            TrainingDataStorage)
else:
    from connect4.neural.nn_tf import ModelWrapper


class TrainingLoop():
    def __init__(self,
                 config: AlphaZeroConfig):
        self.config = config
        self.save_dir = config.storage_config.save_dir

        # if there are existing dirs, load the latest
        subfolders = [f.name for f in os.scandir(self.save_dir) if f.is_dir()]
        if subfolders:
            self.gen = max([int(n) for n in subfolders])
            net_path = self.folder_path + '/net.pth'
            print("Loading Network saved in file {}".format(net_path))
            self.model = ModelWrapper(config.model_config, net_path)
            self.gen += 1
        else:
            self.gen = 1
            self.model = ModelWrapper(config.model_config)

        self.data_storage = TrainingDataStorage()

        if os.path.exists(self.save_dir + '/8ply.pkl'):
            self.stats_8ply = pd.read_pickle(self.save_dir + '/8ply.pkl')
        else:
            self.stats_8ply = pd.DataFrame()
        if os.path.exists(self.save_dir + '/7ply.pkl'):
            self.stats_7ply = pd.read_pickle(self.save_dir + '/7ply.pkl')
        else:
            self.stats_7ply = pd.DataFrame()

        self.opponent = MCTS("mcts:" + str(config.simulations),
                             MCTSConfig(simulations=config.simulations),
                             evl.Evaluator(evl.evaluate_centre_with_prior))

        if os.path.exists(self.save_dir + '/match_results.pkl'):
            self.match_results = pd.read_pickle(
                self.save_dir + '/match_results.pkl')
        else:
            self.match_results = pd.DataFrame(
                columns=['wins', 'draws', 'losses', 'return'])

        if config.visdom_enabled:
            self.vis = Visdom()
            fig = plt.figure()
            self.match_win = self.vis.matplot(fig)
            self.win_8ply = self.vis.matplot(fig)
            self.game_win = self.vis.text('')

    @property
    def folder_path(self):
        return "{}/{}".format(self.save_dir, self.gen)

    def run(self):
        while True:
            print("Loop: ", self.gen)
            self.loop()
            if self.gen % self.config.n_eval == 0:
                self.evaluate()

    def loop(self):
        os.makedirs(self.folder_path)
        start_t = time.time()
        print('Time now: {}'.format(time.asctime(time.localtime(start_t))))
        self.generate_games()
        train_t = time.time()
        self.train()
        end_t = time.time()
        print('Generate games: {:.0f}s  training: {:.0f}s'.format(
            train_t - start_t,
            end_t - train_t))
        self.gen += 1

    def generate_games(self):
        mcts_config = self.create_alpha_zero_config(training=True)
        games = []

        if self.config.game_processes == 1:
            evaluator = evl.Evaluator(partial(evl.evaluate_nn,
                                              model=self.model))
            alpha_zero = MCTS('AlphaZero',
                              mcts_config,
                              evaluator)
            for _ in range(self.config.n_training_games):
                game_data = training_game(alpha_zero)
                games.append(game_data)
        else:
            connections = [[Pipe() for
                            _ in range(self.config.game_threads)] for
                           _ in range(self.config.game_processes)]

            inference_server = InferenceServer(self.model,
                                               [item[1]
                                                for sublist in connections
                                                for item in sublist])

            with Pool(processes=self.config.game_processes) as pool:
                # pool.map(
                for game_batch in pool.imap_unordered(partial(
                        game_pool,
                        mcts_config=mcts_config,
                        n_threads=self.config.game_threads,
                        n_games=int(self.config.n_training_games /
                                    self.config.game_processes)),
                                                      connections,
                                                      chunksize=1):
                    games.extend(game_batch)

            inference_server.terminate()

        self.data_storage.save(games, self.folder_path)

        results = list(map(lambda x: x.result, games))
        print('Player one: wins, draws, losses:  {}, {}, {}'.format(
            results.count(Result.o_win),
            results.count(Result.draw),
            results.count(Result.x_win)))

        if self.config.visdom_enabled:
            self.vis.text(self.data_storage.last_game_str(),
                          win=self.game_win)

    def train(self):
        training_data = self.data_storage.get_dataset(self.save_dir, self.gen)

        print("{} positions created for training".format(len(training_data)))

        self.model.train(training_data, True)
        self.model.save(self.folder_path)

    def evaluate(self):
        ply8_data = Connect4Dataset.load(
            '/home/richard/data/connect4/connect4dataset_8ply.pth')
        value_stats = self.model.evaluate_value_only(ply8_data)
        print("8 Ply Test Stats:  ", value_stats)
        self.stats_8ply = self.stats_8ply.append(value_stats.to_dict(),
                                                 ignore_index=True)
        self.stats_8ply.to_pickle(self.save_dir + '/8ply.pkl')

        ply7_data = Connect4Dataset.load(
            '/home/richard/data/connect4/connect4dataset_7ply.pth')
        combined_stats = self.model.evaluate(ply7_data)
        print("7 Ply Test Stats:  ", combined_stats)
        self.stats_7ply = self.stats_7ply.append(combined_stats.to_dict(),
                                                 ignore_index=True)
        self.stats_7ply.to_pickle(self.save_dir + '/7ply.pkl')

        if self.config.visdom_enabled:
            self.vis.matplot(self.stats_8ply.plot(y=['Accuracy']).figure,
                             win=self.win_8ply)

        az_config = self.create_alpha_zero_config(training=False)
        evaluator = evl.Evaluator(partial(evl.evaluate_nn,
                                          model=self.model))
        alpha_zero = MCTS('AlphaZero',
                          az_config,
                          evaluator)

        results = self.match(alpha_zero, self.opponent)
        self.match_results = self.match_results.append(results,
                                                       ignore_index=True)
        self.match_results.to_pickle(
            self.save_dir + '/match_results.pkl')
        if self.config.visdom_enabled:
            self.vis.matplot(self.match_results.plot(y=['return']).figure,
                             win=self.match_win)

    def match(self, alpha_zero, opponent: MCTS):
        match = Match(False, alpha_zero, opponent, plies=1, switch=True)
        return match.play(agents=max(self.config.game_processes, 5))

    def create_alpha_zero_config(self, training=False):
        if training:
            return MCTSConfig(self.config.simulations,
                              self.config.pb_c_base,
                              self.config.pb_c_init,
                              self.config.root_dirichlet_alpha,
                              self.config.root_exploration_fraction,
                              self.config.num_sampling_moves)
        else:
            return MCTSConfig(self.config.simulations,
                              self.config.pb_c_base,
                              self.config.pb_c_init,
                              0.0,
                              0.0,
                              0)

from connect4.board_c import Board
import connect4.evaluators as evl
from connect4.grid_search import GridSearch
from connect4.match import Match
from connect4.mcts import MCTS, MCTSConfig
from connect4.utils import Result

from connect4.neural.config import AlphaZeroConfig
from connect4.neural.game_pool import game_pool
from connect4.neural.inference_server import InferenceServer
from connect4.neural.storage import NetworkStorage
from connect4.neural.training_game import training_game

from functools import partial
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from torch.multiprocessing import (Manager,
                                   Pipe,
                                   Pool)
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

        # create directories to save in
        os.makedirs(self.save_dir + '/net', exist_ok=True)
        os.makedirs(self.save_dir + '/games', exist_ok=True)
        os.makedirs(self.save_dir + '/stats', exist_ok=True)
        os.makedirs(self.save_dir + '/data', exist_ok=True)

        self.nn_storage = NetworkStorage(self.save_dir,
                                         config.model_config,
                                         ModelWrapper)
        self.data_storage = TrainingDataStorage(self.save_dir)

        if os.path.exists(self.save_dir + '/stats/8ply.pkl'):
            self.stats_8ply = pd.read_pickle(self.save_dir + '/stats/8ply.pkl')
        else:
            self.stats_8ply = pd.DataFrame()
        if os.path.exists(self.save_dir + '/stats/8ply.pkl'):
            self.stats_7ply = pd.read_pickle(self.save_dir + '/stats/8ply.pkl')
        else:
            self.stats_7ply = pd.DataFrame()

        self.opponent = MCTS("mcts:" + str(config.simulations),
                             MCTSConfig(simulations=config.simulations),
                             evl.Evaluator(evl.evaluate_centre_with_prior))

        if os.path.exists(self.save_dir + '/stats/match_results.pkl'):
            self.match_results = pd.read_pickle(
                self.save_dir + '/stats/match_results.pkl')
        else:
            self.match_results = pd.DataFrame(
                columns=['win', 'draw', 'loss', 'return'])

        if config.visdom_enabled:
            self.vis = Visdom()
            fig = plt.figure()
            self.easy_win = self.vis.matplot(fig)
            self.hard_win = self.vis.matplot(fig)
            self.win_8ply = self.vis.matplot(fig)
            self.game_win = self.vis.text('')

    def run(self):
        i = 0
        while True:
            i += 1
            print("Loop: ", i)
            self.loop()
            if i % self.config.n_eval == 0:
                self.evaluate()

    def loop(self):
        model = self.nn_storage.get_model()
        mcts_config = self.create_alpha_zero_config(training=True)

        start_t = time.time()
        print('Time now: {}'.format(time.asctime(time.localtime(start_t))))
        if self.config.game_processes == 1:
            games = []
            evaluator = evl.Evaluator(partial(evl.evaluate_nn,
                                              model=model))
            alpha_zero = MCTS('AlphaZero',
                              mcts_config,
                              evaluator)
            for _ in range(self.config.n_training_games):
                game_data = training_game(alpha_zero).play()
                # N.B. ideally these would be saved inside play(), but...
                # multiprocessing?
                games.apend(game_data)
        else:
            connections = [[Pipe() for
                            _ in range(self.config.game_threads)] for
                           _ in range(self.config.game_processes)]

            inference_server = InferenceServer(model,
                                               [item[1]
                                                for sublist in connections
                                                for item in sublist])

            mgr = Manager()
            games = mgr.list()

            with Pool(processes=self.config.game_processes) as pool:
                # pool.imap_unordered(
                pool.map(
                    partial(game_pool,
                            mcts_config=mcts_config,
                            n_threads=self.config.game_threads,
                            n_games=int(self.config.n_training_games /
                                        self.config.game_processes),
                            games=games),
                    connections,
                    chunksize=1)

            inference_server.terminate()

        self.data_storage.save(games)
        if self.config.visdom_enabled:
            self.vis.text(self.data_storage.last_game_str(),
                          win=self.game_win)
        train_t = time.time()

        training_data = self.data_storage.get_dataset()

        print("{} positions created for training".format(len(training_data)))

        self.nn_storage.train(training_data)

        end_t = time.time()
        print('Generate games: {:.0f}s  training: {:.0f}s'.format(train_t - start_t, end_t - train_t))
        results = list(map(lambda x: x.result, games))
        print('Player one: wins, draws, losses:  {}, {}, {}'.format(
            results.count(Result.o_win),
            results.count(Result.draw),
            results.count(Result.x_win)))

    def evaluate(self):
        model = self.nn_storage.get_model()

        ply8_data = Connect4Dataset.load(
            '/home/richard/data/connect4/connect4dataset_8ply.pth')
        value_stats = model.evaluate_value_only(ply8_data)
        print("8 Ply Test Stats:  ", value_stats)
        self.stats_8ply = self.stats_8ply.append(value_stats.to_dict(),
                                                 ignore_index=True)
        self.stats_8ply.to_pickle(self.save_dir + '/stats/8ply.pkl')

        ply7_data = Connect4Dataset.load(
            '/home/richard/data/connect4/connect4dataset_7ply.pth')
        combined_stats = model.evaluate(ply7_data)
        print("7 Ply Test Stats:  ", combined_stats)
        self.stats_7ply = self.stats_7ply.append(combined_stats.to_dict(),
                                                 ignore_index=True)
        self.stats_7ply.to_pickle(self.save_dir + '/stats/7ply.pkl')

        if self.config.visdom_enabled:
            self.vis.matplot(self.stats_8ply.plot(y=['Accuracy']).figure,
                             win=self.win_8ply)

        az_config = self.create_alpha_zero_config(training=False)
        evaluator = evl.Evaluator(partial(evl.evaluate_nn,
                                          model=model))
        alpha_zero = MCTS('AlphaZero',
                          az_config,
                          evaluator)

        results = self.match(alpha_zero, self.opponent)
        self.match_results = self.match_results.append(results,
                                                       ignore_index=True)
        self.match_results.to_pickle(
            self.save_dir + '/stats/match_results.pkl')
        if self.config.visdom_enabled:
            self.vis.matplot(self.match_results.plot(y=['return']).figure,
                             win=self.easy_win)

    def match(self, alpha_zero, opponent: MCTS):
        match = Match(False, alpha_zero, opponent, plies=1, switch=True)
        return match.play(agents=self.config.game_processes)

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

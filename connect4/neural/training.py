from connect4.board import Board
import connect4.evaluators as evl
from connect4.grid_search import GridSearch
from connect4.match import Match
from connect4.mcts import MCTS, MCTSConfig
from connect4.utils import Result

from connect4.neural.config import AlphaZeroConfig
from connect4.neural.game_pool import game_pool
from connect4.neural.inference_server import InferenceServer
from connect4.neural.storage import (GameStorage,
                                     NetworkStorage)
from connect4.neural.training_game import TrainingData, training_game

from functools import partial
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import time
from torch.multiprocessing import (Manager,
                                   Pipe,
                                   Pool)
from visdom import Visdom

if True:
    from connect4.neural.nn_pytorch import (Connect4Dataset,
                                            ModelWrapper,
                                            native_to_pytorch,
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

        self.nn_storage = NetworkStorage(self.save_dir + '/net',
                                         config.model_config,
                                         ModelWrapper)
        self.game_storage = GameStorage(self.save_dir + '/games')
        self.training_data_storage = TrainingDataStorage(self.save_dir + '/data')

        # self.boards = torch.load(config.storage_config.path_8ply_boards)
        # self.values = torch.load(config.storage_config.path_8ply_values)
        with open('/home/richard/data/connect4/8ply_boards.pkl', 'rb') as f:
            ply8_boards = pickle.load(f)
        with open('/home/richard/data/connect4/8ply_values.pkl', 'rb') as f:
            ply8_values = pickle.load(f)

        self.ply8_data = Connect4Dataset(
            *native_to_pytorch(ply8_boards,
                               ply8_values,
                               add_fliplr=True))

        with open('/home/richard/data/connect4/7ply_boards.pkl', 'rb') as f:
            ply7_boards = pickle.load(f)
        with open('/home/richard/data/connect4/7ply_values.pkl', 'rb') as f:
            ply7_values = pickle.load(f)
        with open('/home/richard/data/connect4/7ply_priors.pkl', 'rb') as f:
            ply7_priors = pickle.load(f)

        self.ply7_data = Connect4Dataset(
            *native_to_pytorch(ply7_boards,
                               ply7_values,
                               ply7_priors,
                               add_fliplr=True))

        self.ply7_data.save('/home/richard/data/connect4/connect4dataset_7ply.pth')
        self.ply8_data.save('/home/richard/data/connect4/connect4dataset_8ply.pth')

        if os.path.exists(self.save_dir + '/stats/8ply.pkl'):
            self.stats_8ply = pd.read_pickle(self.save_dir + '/stats/8ply.pkl')
        else:
            self.stats_8ply = pd.DataFrame()
        if os.path.exists(self.save_dir + '/stats/8ply.pkl'):
            self.stats_7ply = pd.read_pickle(self.save_dir + '/stats/8ply.pkl')
        else:
            self.stats_7ply = pd.DataFrame()

        self.easy_opponent = GridSearch("gridsearch:4",
                                        4,
                                        evl.Evaluator(evl.evaluate_centre))
        self.hard_opponent = MCTS("mcts:" + str(config.simulations),
                                  MCTSConfig(simulations=config.simulations,
                                             pb_c_init=9999),
                                  evl.Evaluator(evl.evaluate_centre_with_prior))

        if os.path.exists(self.save_dir + '/stats/easy_results.pkl'):
            self.easy_results = pd.read_pickle(self.save_dir + '/stats/easy_results.pkl')
        else:
            self.easy_results = pd.DataFrame(columns=['win', 'draw', 'loss', 'return'])

        if os.path.exists(self.save_dir + '/stats/hard_results.pkl'):
            self.hard_results = pd.read_pickle(self.save_dir + '/stats/hard_results.pkl')
        else:
            self.hard_results = pd.DataFrame(columns=['win', 'draw', 'loss', 'return'])

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
            self.loop(i)
            if i % self.config.n_eval == 0:
                self.evaluate()

    def loop(self, iteration):
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

        if self.config.visdom_enabled:
            self.vis.text(self.game_storage.last_game_str(),
                          win=self.game_win)
        self.game_storage.save(games)
        train_t = time.time()

        training_data = self.training_data_storage.get_dataset(
            self.calc_n_posn(iteration),
            games)

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

        value_stats = model.evaluate_value_only(self.ply8_data)
        print("8 Ply Test Stats:  ", value_stats)
        self.stats_8ply = self.stats_8ply.append(value_stats.to_dict(),
                                                 ignore_index=True)
        self.stats_8ply.to_pickle(self.save_dir + '/stats/8ply.pkl')

        combined_stats = model.evaluate(self.ply7_data)
        print("7 Ply Test Stats:  ", combined_stats)
        self.stats_7ply = self.stats_7ply.append(combined_stats.to_dict(),
                                                 ignore_index=True)
        self.stats_7ply.to_pickle(self.save_dir + '/stats/7ply.pkl')

        if self.config.visdom_enabled:
            self.vis.matplot(self.stats_8ply.plot(y=['Accuracy']).figure,
                             win=self.win_8ply)

        # alpha_zero = self.create_alpha_zero(training=False)

        # results = self.match(alpha_zero, self.easy_opponent)
        # self.easy_results = self.easy_results.append(results, ignore_index=True)
        # self.easy_results.to_pickle(self.save_dir + '/stats/easy_results.pkl')
        # if self.config.visdom_enabled:
        #     self.vis.matplot(self.easy_results.plot(y=['return']).figure,
        #                      win=self.easy_win)

        # results = self.match(alpha_zero, self.hard_opponent)
        # self.hard_results = self.hard_results.append(results, ignore_index=True)
        # self.hard_results.to_pickle(self.save_dir + '/stats/hard_results.pkl')
        # if self.config.visdom_enabled:
        #     self.vis.matplot(self.hard_results.plot(y=['return']).figure, win=self.hard_win)

    def match(self, alpha_zero, opponent: MCTS):
        match = Match(False, alpha_zero, opponent, plies=1, switch=True)
        return match.play(agents=self.config.agents)

    def create_alpha_zero_config(self, training=False):
        if training:
            mcts_config = MCTSConfig(self.config.simulations,
                                     self.config.pb_c_base,
                                     self.config.pb_c_init,
                                     self.config.root_dirichlet_alpha,
                                     self.config.root_exploration_fraction,
                                     self.config.num_sampling_moves)
        else:
            mcts_config = MCTSConfig(self.config.simulations,
                                     self.config.pb_c_base,
                                     self.config.pb_c_init,
                                     0.0,
                                     0.0,
                                     0)

        return mcts_config

    def calc_n_posn(self, iteration):
        n = min(20, iteration / 2)
        return n * 25 * 2 * self.config.n_training_games

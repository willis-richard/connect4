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
                                     NetworkStorage,
                                     ReplayStorage)
from connect4.neural.training_game import training_game

from functools import partial
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import torch
from visdom import Visdom


class TrainingLoop():
    def __init__(self,
                 config: AlphaZeroConfig):
        self.config = config
        self.save_dir = config.storage_config.save_dir

        # create directories to save in
        os.makedirs(self.save_dir + '/net', exist_ok=True)
        os.makedirs(self.save_dir + '/games', exist_ok=True)
        os.makedirs(self.save_dir + '/stats', exist_ok=True)

        if config.use_pytorch:
            from connect4.neural.nn_pytorch import ModelWrapper
        else:
            from connect4.neural.nn_tf import ModelWrapper

        self.nn_storage = NetworkStorage(self.save_dir + '/net',
                                         config.model_config,
                                         ModelWrapper)
        self.replay_storage = ReplayStorage()
        self.game_storage = GameStorage(self.save_dir + '/games')

        self.boards = torch.load(config.storage_config.path_8ply_boards)
        self.values = torch.load(config.storage_config.path_8ply_values)

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

        if os.path.exists(self.save_dir + '/stats/8ply.pkl'):
            self.stats_8ply = pd.read_pickle(self.save_dir + '/stats/8ply.pkl')
        else:
            self.stats_8ply = pd.DataFrame()

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
            if i % self.config.n_eval == 1:
                self.evaluate()

    def loop(self):
        model = self.nn_storage.get_model()
        mcts_config = self.create_alpha_zero_config(training=True)
        self.replay_storage.reset()
        game_results = []

        start_t = time.time()
        if self.config.game_processes == 1:
            if self.config.agents == 1:
                evaluator = evl.Evaluator(partial(evl.evaluate_nn,
                                                  model=model))
                alpha_zero = MCTS('AlphaZero',
                                  mcts_config,
                                  evaluator)
            for _ in range(self.config.n_training_games):
                result, history, boards, values, policies = training_game(alpha_zero).play()
                # N.B. ideally these would be saved inside play(), but...
                # multiprocessing?
                game_results.append(result)
                self.replay_storage.save_game(boards, values, policies)
                self.game_storage.save_game(history)
        else:
            from torch.multiprocessing import (Pipe,
                                               Pool,
                                               set_start_method)
            try:
                set_start_method('spawn')
            except RuntimeError as e:
                if str(e) == 'context has already been set':
                    pass

            connections = [[Pipe() for
                            _ in range(self.config.game_threads)] for
                           _ in range(self.config.game_processes)]

            inference_server = InferenceServer(model,
                                               [item[1] for sublist in connections for item in sublist])

            game_pool_args = [(mcts_config,
                               self.config.game_threads,
                               conns,
                               self.config.n_training_games / self.config.game_processes) for
                              conns in connections]

            with Pool(processes=self.config.game_processes) as pool:
                results = pool.starmap(game_pool, game_pool_args)
            for result, history, data in results:
                self.replay_storage.save_game(*data)
                self.game_storage.save_game(history)
                game_results.append(result)

        if self.config.visdom_enabled:
            self.vis.text(self.game_storage.last_game_str(),
                          win=self.game_win)
        self.game_storage.save()
        train_t = time.time()

        self.nn_storage.train(self.replay_storage.get_data(),
                              self.config.n_training_epochs)
        end_t = time.time()
        print('Time now: {}'.format(time.asctime(time.localtime(end_t))))
        print('Generate games: {:.0f}s  training: {:.0f}s'.format(train_t - start_t, end_t - train_t))
        print('Player one: wins, draws, losses:  {}, {}, {}'.format(
            game_results.count(Result.o_win),
            game_results.count(Result.draw),
            game_results.count(Result.x_win)))

    def evaluate(self):
        model = self.nn_storage.get_model()

        test_stats = model.evaluate_value_only(self.boards, self.values)

        print("8 Ply Test Stats:  ", test_stats)
        self.stats_8ply = self.stats_8ply.append(test_stats.to_dict(), ignore_index=True)
        self.stats_8ply.to_pickle(self.save_dir + '/stats/8ply.pkl')

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

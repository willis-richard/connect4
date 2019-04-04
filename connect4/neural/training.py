from connect4.board import Board
import connect4.evaluators as e
from connect4.grid_search import GridSearch
from connect4.match import Match
from connect4.mcts import MCTS, MCTSConfig
from connect4.player import BasePlayer
from connect4.utils import Result

from connect4.neural.config import AlphaZeroConfig
from connect4.neural.storage import (GameStorage,
                                     NetworkStorage,
                                     ReplayStorage)

from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.utils.data as data
from visdom import Visdom


def top_level_defined_play(x):
    return TrainingGame(x).play()

class TrainingGame():
    def __init__(self, player: BasePlayer):
        self.player = player

    def play(self):
        board = Board()
        boards = []
        values = []
        policies = []
        history = []
        while board.result is None:
            move, value, tree = self.player.make_move(board)

            boards.append(board.to_array())
            values.append(value)
            policy = tree.get_policy()
            policies.append(policy)

            history.append((move, value, policy))

        values = self.create_values(values, board.result)

        return board.result, history, (boards, values, policies)

    def create_values(self, mcts_values, result):
        length = len(values)
        # label board data with result
        result_values = np.ones((length,), dtype=np.float64)
        if result == Result.o_win:
            result_values[1::2] = 0.0
        elif result == Result.x_win:
            result_values[0::2] = 0.0
        else:
            result_values *= 0.5
        merged_values = (np.array(mcts_values) + result_values) / 2.0
        return merged_values


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
                                        e.Evaluator(e.evaluate_centre))
        self.hard_opponent = MCTS("mcts:2500",
                                  MCTSConfig(simulations=2500,
                                             pb_c_init=9999),
                                  e.Evaluator(e.evaluate_centre_with_prior))

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
        alpha_zero = self.create_alpha_zero(training=True)

        self.replay_storage.reset()
        game_results = []

        import time
        start_t = time.time()
        if self.config.agents == 1:
            for _ in range(self.config.n_training_games):
                result, history, data = TrainingGame(alpha_zero).play()
                # N.B. ideally these would be saved inside play(), but... multiprocess?
                game_results.append(result)
                self.replay_storage.save_game(*data)
                self.game_storage.save_game(history)
        else:
            from torch.multiprocessing import Pool, Process, set_start_method
            try:
                set_start_method('spawn')
            except RuntimeError as e:
                if str(e) == 'context has already been set':
                    pass

            a0 = [alpha_zero for _ in range(self.config.n_training_games)]
            with Pool(processes=self.config.agents) as pool:
                results = pool.map(top_level_defined_play, a0)
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

        alpha_zero = self.create_alpha_zero(training=False)

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

    def create_alpha_zero(self, training=False):
        model = self.nn_storage.get_model()
        evaluator = e.NetEvaluator(
            e.evaluate_nn,
            model)
        player = MCTS('AlphaZero',
                      MCTSConfig(self.config.simulations,
                                 self.config.pb_c_init,
                                 self.config.root_dirichlet_alpha if training else 0.0,
                                 self.config.root_exploration_fraction if training else 0.0),
                      evaluator)
        return player

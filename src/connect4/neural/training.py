from src.connect4.board import Board
import src.connect4.evaluators as e
from src.connect4.grid_search import GridSearch
from src.connect4.match import Match
from src.connect4.mcts import MCTS, MCTSConfig
from src.connect4.player import BasePlayer
from src.connect4.utils import Result

from src.connect4.neural.config import AlphaZeroConfig
from src.connect4.neural.storage import (Connect4Dataset,
                                         GameStorage,
                                         NetworkStorage,
                                         ReplayStorage)

from copy import copy
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
        policies = []
        history = []
        while board.result is None:
            move, value, tree = self.player.make_move(board)

            history.append((move, value))

            boards.append(board.to_tensor())
            policy = tree.get_policy()
            policies.append(torch.tensor(policy))

        boards = torch.stack(boards).squeeze()
        values = self.create_values(board.result, len(boards))
        policies = torch.stack(policies)

        return board.result, history, (boards, values, policies)

    def create_values(self, result, length):
        # label board data with result
        values = torch.ones((length,), dtype=torch.float)
        if result == Result.o_win:
            values[1::2] = 0.
        elif result == Result.x_win:
            values[0::2] = 0.
        else:
            values *= 0.5
        return values


class TrainingLoop():
    def __init__(self,
                 config: AlphaZeroConfig):
        self.config = config
        self.save_dir = config.storage_config.save_dir

        # create directories to save in
        os.makedirs(self.save_dir + '/net', exist_ok=True)
        os.makedirs(self.save_dir + '/games', exist_ok=True)
        os.makedirs(self.save_dir + '/stats', exist_ok=True)

        self.nn_storage = NetworkStorage(self.save_dir + '/net',
                                         config.model_config)
        self.replay_storage = ReplayStorage(config.model_config)
        self.game_storage = GameStorage(self.save_dir + '/games')

        boards = torch.load(config.storage_config.path_8ply_boards)
        values = torch.load(config.storage_config.path_8ply_values)

        # Note no policy here, 3rd arg unused
        test_data = Connect4Dataset(boards, values, values)
        self.test_data = data.DataLoader(test_data, batch_size=4096)

        self.easy_opponent = GridSearch("gridsearch:4",
                                        4,
                                        e.Evaluator(e.evaluate_centre))
        self.hard_opponent = MCTS("mcts:2500",
                                  MCTSConfig(simulations=2500,
                                             pb_c_init=9999),
                                  e.Evaluator(e.evaluate_centre_with_prior))

        if os.path.exists(self.save_dir + '/stats/easy_results.pkl'):
            self.stats = pd.read_pickle(self.save_dir + '/stats/easy_results.pkl')
        else:
            self.easy_results = pd.DataFrame(columns=['win', 'draw', 'loss', 'return'])

        if os.path.exists(self.save_dir + '/stats/hard_results.pkl'):
            self.stats = pd.read_pickle(self.save_dir + '/stats/hard_results.pkl')
        else:
            self.hard_results = pd.DataFrame(columns=['win', 'draw', 'loss', 'return'])

        if os.path.exists(self.save_dir + '/stats/8ply.pkl'):
            self.stats = pd.read_pickle(self.save_dir + '/stats/8ply.pkl')
        else:
            self.stats_8ply = pd.DataFrame()

        if config.visdom_enabled:
            self.vis = Visdom()
            self.easy_win = self.vis.matplot()
            self.hard_win = self.vis.matplot()
            self.win_8ply = self.vis.matplot()
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

        import time
        start = time.time()
        if self.config.agents == 1:
            for _ in range(self.config.n_training_games):
                _, history, data = TrainingGame(alpha_zero).play()
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
            for _, history, data in results:
                self.replay_storage.save_game(*data)
                self.game_storage.save_game(history)

        if self.config.visdom_enabled:
            self.vis.text(str(self.game_storage.games[-1]), win=self.game_win)
        self.game_storage.save()
        train = time.time()

        self.nn_storage.train(self.replay_storage.get_data(),
                              self.config.n_training_epochs)
        end = time.time()
        print('Generate games: {:.0f}  training: {:.0f}'.format(train - start, end - train))

    def evaluate(self):
        self.test_8ply()

        if self.config.visdom_enabled:
            self.vis.matplot(self.stats_8ply.plot(y=['Accuracy']), win=self.win_8ply)

        alpha_zero = self.create_alpha_zero(training=False)

        results = self.match(alpha_zero, self.easy_opponent)
        self.easy_results = self.easy_results.append(results, ignore_index=True)
        self.easy_results.to_pickle(self.save_dir + '/stats/easy_results.pkl')
        if self.config.visdom_enabled:
            self.vis.matplot(self.easy_results.plot(y=['return']), win=self.easy_win)

        # results = self.match(alpha_zero, self.hard_opponent)
        # self.hard_results = self.hard_results.append(results, ignore_index=True)
        # self.hard_results.to_pickle(self.save_dir + '/stats/hard_results.pkl')
        # if self.config.visdom_enabled:
        #     self.vis.matplot(self.hard_results.plot(y=['return']), win=self.hard_win)

    def test_8ply(self):
        """Get an idea of how the initialisation is"""
        test_stats = Stats()
        model = self.nn_storage.get_model()
        net = model.net
        device = model.device
        criterion = model.value_loss

        with torch.set_grad_enabled(False):
            net = net.eval()
            for board, value, _ in self.test_data:
                board, y_value = board.to(device), value.to(device)
                x_value, _ = net(board)
                loss = criterion(x_value, y_value)
                test_stats.update(x_value, y_value, loss)

        print("8 Ply Test Stats:  ", test_stats)
        self.stats_8ply = self.stats_8ply.append(test_stats.to_dict(), ignore_index=True)
        self.stats_8ply.to_pickle(self.save_dir + '/stats/8ply.pkl')

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


def categorise_predictions(preds):
    preds = preds * 3.0
    torch.floor_(preds)
    preds = preds / 2.0
    return preds


class Stats():
    def __init__(self):
        self.n = 0
        self.average_value = 0.0
        self.total_loss = 0.0
        self.smallest = 1.0
        self.largest = 0.0
        self.correct = {i: 0 for i in [0.0, 0.5, 1.0]}
        self.total = {i: 0 for i in [0.0, 0.5, 1.0]}

    @property
    def loss(self):
        return self.total_loss / self.n

    @property
    def accuracy(self):
        return float(sum(self.correct.values())) / self.n

    @property
    def average(self):
        return self.average_value / self.n

    def to_dict(self):
        dict_ = {'Average loss': self.loss,
                 'Accuracy': self.accuracy,
                 'Smallest':self.smallest,
                 'Largest': self.largest,
                 'Average': self.average}
        dict_['correct'] = {}
        for k in self.correct:
            dict_['correct'][k] = (self.total[k], self.correct[k])

        return dict_

    def __repr__(self):
        x = "Average loss:  " + "{:.5f}".format(self.loss) + \
            "  Accuracy:  " + "{:.5f}".format(self.accuracy) + \
            "  Smallest:  " + "{:.5f}".format(self.smallest) + \
            "  Largest:  " + "{:.5f}".format(self.largest) + \
            "  Average:  " + "{:.5f}".format(self.average) + \
            "\nCategory, # Members, # Correct Predictions:"

        for k in self.correct:
            x += "  ({}, {}, {})".format(
                k,
                self.total[k],
                self.correct[k])
        return x

    def update(self, outputs, values, loss):
        self.n += len(values)
        self.average_value += outputs.sum().item()
        self.total_loss += loss.item() * len(values)
        self.smallest = min(self.smallest, torch.min(outputs).item())
        self.largest = max(self.largest, torch.max(outputs).item())

        categories = categorise_predictions(outputs)
        values = values.view(-1)
        categories = categories.view(-1)

        for k in self.correct:
            idx = (values == k).nonzero()
            self.total[k] += len(idx)
            self.correct[k] += len(torch.eq(categories[idx], values[idx]).nonzero())

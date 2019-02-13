from src.connect4.board import Board
import src.connect4.evaluators as evaluators
from src.connect4.match import Match
from src.connect4.player import BasePlayer
from src.connect4.mcts import MCTS
from src.connect4.utils import Result

from src.connect4.neural.config import AlphaZeroConfig
from src.connect4.neural.storage import (Connect4Dataset,
                                         NetworkStorage,
                                         ReplayStorage)

from copy import copy
import torch
import torch.utils.data as data


class TrainingGame():
    def __init__(self,
                 player: BasePlayer,
                 replay_storage: ReplayStorage):
        self.player = player
        self.replay_storage = replay_storage

    def play(self):
        board = Board()
        boards = []
        policies = []
        while self.board.result is None:
            self.player.make_move(board)

            boards.append(self.board.to_tensor())
            policy = self.player.tree.get_policy()
            policies.append(torch.tensor(policy))

        boards = torch.stack(boards).squeeze()
        values = self.create_values(len(boards))
        policies = torch.stack(policies)

        self.replay_storage.save_game(boards,
                                      values,
                                      policies)
        return self.board.result

    def create_values(self, length):
        # label board data with result
        values = torch.ones((length,), dtype=torch.float)
        if self.board.result == Result.o_win:
            values[1::2] = 0.
        elif self.board.result == Result.x_win:
            values[0::2] = 0.
        else:
            values *= 0.5
        return values


class TrainingLoop():
    def __init__(self,
                 config: AlphaZeroConfig,
                 folder_path: str):
        self.config = config
        self.nn_storage = NetworkStorage(folder_path)
        self.replay_storage = ReplayStorage(config, folder_path)

        boards = torch.load('/home/richard/Downloads/connect4_boards.pth')
        values = torch.load('/home/richard/Downloads/connect4_values.pth')

        # Note no policy here, 3rd arg unused
        test_data = Connect4Dataset(boards, values, values)
        self.test_data = data.DataLoader(test_data, batch_size=4096)

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
        evaluator = evaluators.NetEvaluator(
            evaluators.evaluate_nn,
            model)
        player = MCTS('AlphaZero',
                      MCTS.Config(simulations=100,
                                  cpuct=9999),
                      evaluator)

        self.replay_storage.reset()

        for _ in range(self.config.n_training_games):
            TrainingGame(copy(player),
                         Board(),
                         self.replay_storage).play()

        self.nn_storage.train(self.replay_storage.get_data(),
                              self.config.n_training_epochs)

    def evaluate(self):
        self.test_8ply()
        evaluator = evaluators.Evaluator(
            evaluators.evaluate_centre_with_prior)
        self.match(MCTS("mcts",
                        MCTS.Config(simulations=100,
                                    cpuct=9999),
                        evaluator))
        self.match(MCTS("mcts",
                        MCTS.Config(simulations=2500,
                                    cpuct=9999),
                        evaluator))
        return

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

    def match(self, opponent: MCTS):
        player = MCTS('AlphaZero',
                                MCTS(MCTS.Config(simulations=100,
                                                 cpuct=9999)),
                                net=self.nn_storage.get_model())

        match = Match(False, player, opponent, plies=2, switch=True)
        # match.play_parallel(agents=self.config.agents)
        match.play()
        return


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

    def __repr__(self):
        x = "Average loss:  " + "{:.5f}".format(self.loss) + \
            "  Accuracy:  " + "{:.5f}".format(self.accuracy) + \
            "  Smallest:  " + "{:.5f}".format(self.smallest) + \
            "  Largest:  " + "{:.5f}".format(self.largest) + \
            "  Average:  " + "{:.5f}".format(self.average) + \
            "\nCategory, # Members, # Correct Predictions:"

        for k in self.correct:
            x += "{}, {}, {}".format(
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

from connect4.evaluators import Evaluator
from connect4.mcts import MCTS, MCTSConfig

from connect4.neural.inference_server import evaluate_server
from connect4.neural.training_game import TrainingData, training_game

from collections import deque
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.pool import ThreadPool
import numpy as np
import os
from typing import Dict, List, Tuple, Optional


def game_pool(mcts_config: MCTSConfig,
              n_threads: int,
              conn_list: List[Tuple[Connection, Connection]],
              n_games: int):
    assert n_threads == len(conn_list)

    position_table = {}
    result_table = {}

    thread_args = [MCTS('AlphaZero:{}:{}'.format(os.getpid(), i),
                        mcts_config,
                        Evaluator(partial(evaluate_server,
                                          conn=conn[0]),
                                  position_table,
                                  result_table,
                                  store_position=False))
                   for i, conn in enumerate(conn_list)]

    results = []
    games = []
    training_data = TrainingData()

    for _ in range(n_games):
        with ThreadPool(n_threads) as pool:
            training_games = pool.map(training_game, thread_args)
            for game_data in training_games:
                results.append(game_data.result)
                games.append(game_data.game)
                training_data.add(game_data.data)

    return results, games, training_data


def run_training_game(player_deque):
    player = player_deque.pop()

    results = training_game(player)

    player_deque.push(player)

    return results

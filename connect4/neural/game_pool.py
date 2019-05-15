from connect4.evaluators import Evaluator
from connect4.mcts import MCTS, MCTSConfig

from connect4.neural.inference_server import evaluate_server, evaluate_server_deque
from connect4.neural.training_game import TrainingData, training_game

from collections import deque
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.pool import ThreadPool
import numpy as np
import os
from typing import Dict, List, Tuple, Optional


def game_pool(conn_list: List[Tuple[Connection, Connection]],
              n_threads: int,
              mcts_config: MCTSConfig,
              n_games: int):
    assert n_threads == len(conn_list)

    position_table = {}
    conn_deque = deque([c[0] for c in conn_list])

    evaluator = Evaluator(partial(evaluate_server_deque,
                                  conn_deque=conn_deque),
                          position_table,
                          store_position=True)

    player_deque = deque([MCTS('AlphaZero:{}:{}'.format(os.getpid(), i),
                               mcts_config,
                               evaluator)
                          for i in range(n_threads)])

    results = []
    games = []
    training_data = TrainingData()
    thread_args = [i for i in range(n_games)]

    with ThreadPool(n_threads) as pool:
        training_games = pool.map(partial(run_training_game,
                                          player_deque=player_deque),
                                  thread_args)
        for game_data in training_games:
            results.append(game_data.result)
            games.append(game_data.game)
            training_data.add(game_data.data)
    return results, games, training_data


def run_training_game(useless_int, player_deque: deque):
    player = player_deque.pop()
    results = training_game(player)
    player_deque.append(player)
    return results

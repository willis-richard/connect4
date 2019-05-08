from connect4.evaluators import Evaluator
from connect4.mcts import MCTS, MCTSConfig

from connect4.neural.inference_server import evaluate_server
from connect4.neural.training_game import training_game

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
                                  store_position=True))
                   for i, conn in enumerate(conn_list)]

    result_list = []
    moves_list = []
    board_list = []
    value_list = []
    policy_list = []

    for _ in range(n_games):
        with ThreadPool(n_threads) as pool:
            # FIXME: USE PARAMETER n_games
            results = pool.map(training_game, thread_args)
            for result, board, move, value, policy in results:
                result_list.append(result)
                board_list.extend(board)
                moves_list.extend(move)
                value_list.extend(value)
                policy_list.extend(policy)

    return result_list, board_list, moves_list, value_list, policy_list


def run_training_game(player_deque):
    player = player_deque.pop()

    results = training_game(player)

    player_deque.push(player)

    return results

from connect4.evaluators import Evaluator
from connect4.mcts import MCTS, MCTSConfig

from connect4.neural.inference_server import evaluate_server
from connect4.neural.training_game import training_game

from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.pool import ThreadPool
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

    result_list = []
    history_list = []
    data_list = []

    with ThreadPool(n_threads) as pool:
        results = pool.map(training_game, thread_args)
        for result, history, data in results:
            result_list.append(result)
            history_list.append(history)
            data_list.append(data)

    return result_list, history_list, data_list

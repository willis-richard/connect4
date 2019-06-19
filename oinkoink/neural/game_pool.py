from oinkoink.evaluators import Evaluator
from oinkoink.mcts import MCTS, MCTSConfig

from oinkoink.neural.inference_server import evaluate_server_deque
from oinkoink.neural.training_game import training_game

from collections import deque
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.pool import ThreadPool
import os
from typing import Dict, List, Tuple


def game_pool(conn_list: List[Tuple[Connection, Connection]],
              n_threads: int,
              mcts_config: MCTSConfig,
              n_games: int):
    assert n_threads == len(conn_list)

    position_table: Dict[Tuple, Tuple] = {}
    conn_deque = deque([c[0] for c in conn_list])

    evaluator = Evaluator(partial(evaluate_server_deque,
                                  conn_deque=conn_deque),
                          position_table,
                          store_position=True)

    player_deque = deque([MCTS('AlphaZero:{}:{}'.format(os.getpid(), i),
                               mcts_config,
                               evaluator)
                          for i in range(n_threads)])

    games = []

    with ThreadPool(n_threads) as pool:
        for game_data in pool.imap_unordered(partial(run_training_game,
                                                     player_deque=player_deque),
                                             range(n_games),
                                             chunksize=1):
            games.append(game_data)
    return games


def run_training_game(useless_int, player_deque: deque):
    player = player_deque.pop()
    results = training_game(player)
    player_deque.append(player)
    return results

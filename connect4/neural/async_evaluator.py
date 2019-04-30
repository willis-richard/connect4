from connect4.board import Board, make_random_ips
from connect4.evaluators import Evaluator
from connect4.utils import Connect4Stats as info

from copy import copy
import datetime as dt
from functools import partial
from torch.multiprocessing import Lock, Pipe, Process
from scipy.special import softmax
from typing import Callable, Dict, List, Optional, Set

import numpy as np

class NetEvaluator(Evaluator):
    def __init__(self,
                 evaluate_fn: Callable,
                 model,
                 position_table: Optional[Dict] = None,
                 result_table: Optional[Dict] = None,
                 initialise_cache_depth: int = 0):
        self.model = model
        super().__init__(partial(evaluate_fn, model=self.model),
                         position_table,
                         result_table)

        if initialise_cache_depth > 0:
            ips = make_random_ips(initialise_cache_depth)
            self.model(list(ips))

        # self.lock = Lock()

        # def __call__(self, board: Board):
        #     with self.lock():
        #         position_eval = super().__call__(board)
        #     return position_eval


# ok so I either create one of these and for each alpha zero training I create a new connection to this instance and they use that to send requests and it communicates back...
# or I use the base manager and make this shared between processes right?
class AsyncNetEvaluator():
    # But I need a process manager to share this right?
    def __init__(self,
                 model,
                 batch_size: int,
                 timeout_seconds: float,
                 max_wait_microseconds: int,
                 position_table: Optional[Dict] = None,
                 result_table: Optional[Dict] = None):
        self.position_table = {}
        self.result_table = {}
        # self.position_lock = Lock()
        # self.result_lock = Lock()
        self.requester = Requester(model,
                                   batch_size,
                                   timeout_seconds,
                                   max_wait_microseconds)
        self.start()

    # @property
    # def result_table(self):
    #     return

    # # err help
    # @result_table.setter
    # def result_table(self, board):
    #     with self.result_lock:
    #         self._result_table[board]

    def __call__(self, board: Board):
        position_eval = None
        # with self.position_lock:
        #     if board in self.position_table:
        #         position_eval = self.position_table[board]
        if board in self.position_table:
            print("Cached posn")
            position_eval = self.position_table[board]

        if position_eval is not None:
            return position_eval

        self.requester_send.send(board)
        # return None
        return 0.5, np.ones((7,))

    def start(self):
        # start the receiver
        listener_receive, listener_send = Pipe(False)
        listener_p = Process(target=AsyncNetEvaluator.listen,
                             args=(listener_receive,
                                   # self.lock,
                                   self.position_table))
        listener_p.start()
        # listener_p.join()

        # start the requester
        requester_receive, self.requester_send = Pipe(False)
        requester_p = Process(target=self.requester.start,
                              args=(requester_receive, listener_send))
        requester_p.start()
        # requester_p.join()

    def stop(self):
        self.requester_send.close()

    @staticmethod
    def listen(conn,
               # lock,
               position_table):
        while True:
            try:
                boards, position_evals = conn.recv()
            except EOFError:
                break
                # with lock:
            #     for board, position_eval in zip(boards, position_evals):
            #         position_table[board] = position_eval
            for board, position_eval in zip(boards, position_evals):
                position_table[board] = position_eval


class Requester():
    def __init__(self,
                 model,
                 batch_size: int,
                 timeout_seconds: float,
                 max_wait_microseconds: int):
        self.model = model
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.max_wait_microseconds = dt.timedelta(microseconds=max_wait_microseconds)
        self.requests: Set[Board] = set()
        self.first_request_t = None

    def start(self, conn, listener_send):
        self.listener_send = listener_send
        while True:
            is_requests = conn.poll(self.timeout_seconds)
            if is_requests:
                try:
                    requests = conn.recv()
                except EOFError:
                    break
                self.requests.add(requests)
                if len(self.requests) >= self.batch_size:
                    self.evaluate()
                elif self.first_request_t is None:
                    self.first_request_t = dt.datetime.now()
            if self.requests and self.check_time():
                self.evaluate()

    def evaluate(self):
        position_eval = self.model(list(self.requests))
        self.listener_send.send((self.requests, position_eval))
        self.requests = set()

    def check_time(self):
        if self.first_request_t and \
           (dt.datetime.now() - self.first_request_t) > self.max_wait_microseconds:
            self.first_request_t = None
            return True
        return False


def evaluate_nn(board: Board,
                model):
    value, prior = model(board)
    # prior = softmax(prior)
    # return value, prior
    prior = copy(info.prior)
    return value, prior

from connect4.board import Board, make_random_ips
from connect4.utils import Connect4Stats as info

from copy import copy
import datetime as dt
import math
from multiprocessing.connection import Connection, wait
import os
from torch.multiprocessing import Process
from scipy.special import softmax
from typing import Dict, List


class InferenceServer():
    def __init__(self,
                 model,
                 timeout_microseconds: int,
                 max_wait_microseconds: int,
                 conn_list: List,
                 initialise_cache_depth: int = 0):
        self.model = model
        self.timeout_seconds = float(timeout_microseconds / 1e6)
        self.max_wait_microseconds = dt.timedelta(
            microseconds=max_wait_microseconds)
        self.conn_list = conn_list

        self.batch_size = math.ceil(len(self.conn_list) * 0.75)
        self.request_conns: List[Connection] = list()
        self.request_boards: List[Board] = list()

        self.start()

    def start(self):
        self.p = Process(target=self.run)
        self.p.start()

    def run(self):
        while True:
            for c in wait(self.conn_list): #, self.timeout_seconds):
                try:
                    board = c.recv()
                except EOFError:
                    self.conn_list.remove(c)
                else:
                    self.request_conns.append(c)
                    self.request_boards.append(board)
            if self.request_boards: # and (len(self.request_boards) >= self.batch_size or self.check_time()):
                self.evaluate()

    def evaluate(self):
        # print("send {} positions to GPU".format(len(self.request_boards)))
        values, priors = self.model(self.request_boards)

        # send the results to the connections
        # FIXME: very dumb lambda here
        # map(lambda x: x.send(x),
        #     [(c, (v, p)) for c, v, p in
        #     zip(self.request_conns, values, priors)])
        for c, v, p in zip(self.request_conns, values, priors):
            c.send((v, p))

        self.request_conns = list()
        self.request_boards = list()
        self.first_request_t = None

    def check_time(self):
        if self.first_request_t is None:
            self.first_request_t = dt.datetime.now()
            return False
        elif (dt.datetime.now() - self.first_request_t) > self.max_wait_microseconds:
            return True
        else:
            return False


def evaluate_server(board: Board, conn: Connection):
    conn.send(board)
    value, prior = conn.recv()
    # FIXME: Temporary until net returns priors
    # priors = softmax(prior)
    prior = copy(info.prior)
    return value, prior

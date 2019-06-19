from oinkoink.board import Board, make_random_ips
from oinkoink.utils import Connect4Stats as info

from collections import deque
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
                 conn_list: List):
        self.model = model
        self.conn_list = conn_list

        self.request_conns: List[Connection] = list()
        self.request_boards: List[Board] = list()

        self.start()

    def start(self):
        self.p = Process(target=self.run)
        self.p.start()

    def close(self):
        self.p.close()

    def terminate(self):
        self.p.terminate()

    def run(self):
        while self.conn_list:
            for c in wait(self.conn_list):
                try:
                    board = c.recv()
                except EOFError:
                    self.conn_list.remove(c)
                else:
                    self.request_conns.append(c)
                    self.request_boards.append(board)
            if self.request_boards:
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


def evaluate_server(board: Board, conn: Connection):
    conn.send(board)
    value, prior = conn.recv()
    return value, prior


def evaluate_server_deque(board: Board, conn_deque: deque):
    conn = conn_deque.pop()
    result = evaluate_server(board, conn)
    conn_deque.append(conn)
    return result

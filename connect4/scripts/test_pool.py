import multiprocessing as m
import os
from time import sleep


def initialiser():
    print("init {}".format(os.getpid()))

def work(i):
    sleep(i % 2)
    print("{} work {}".format(i, os.getpid()))

n = [i for i in range(20)]

# with m.Pool(processes=2,
#             initializer=initialiser,
#             maxtasksperchild=5) as pool:
#     pool.map(work, n, chunksize=1)


cts = m.get_context()
class CustomProcess(cts.Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("init {}".format(os.getpid()))
        print(self._args)
        print(self._target)
        self.memory = 0

    def do_my_thing(self):
        self.memory += 1
        print("do my thing {}".format(self.memory))

    def run(self):
        self.do_my_thing()
        super().run()

cts.Process = CustomProcess

with cts.Pool(processes=2,
              maxtasksperchild=5) as pool:
    pool.map(work, n, chunksize=1)

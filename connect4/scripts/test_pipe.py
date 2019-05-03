import multiprocessing
import time


class Consumer(multiprocessing.Process):
    def __init__(self, conn):
        multiprocessing.Process.__init__(self)
        self.conn = conn

    def run(self):
        while True:
            received = self.conn.recv()
            if received is None:
                break


if __name__ == '__main__':
    c1, c2 = multiprocessing.Pipe()

    data = [i for i in range((int(1e6)))]
    data_2 = [[i for i in range(100)] for _ in range(int(1e4))]

    consumer = Consumer(c2)
    consumer.start()
    time.sleep(1)

    start = time.time()

    while data:
        c1.send(data.pop())
    c1.send(None)

    consumer.join()

    end = time.time()
    print("time taken {}s".format(end - start))

    consumer_2 = Consumer(c2)
    consumer_2.start()
    time.sleep(1)

    start = time.time()

    while data_2:
        c1.send(data_2.pop())
    c1.send(None)

    consumer_2.join()

    end = time.time()
    print("time taken {}s".format(end - start))

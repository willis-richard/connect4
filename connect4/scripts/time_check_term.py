from connect4.board_c import Board

import pickle
import time

if __name__ == '__main__':
    # with open('/home/richard/data/connect4/7ply_ips.pkl', 'rb') as f:
    #     board_ips = pickle.load(f)
    board = Board()

    start_t = time.time()
    for i in range(int(1e5)):
        board.check_for_winner(board.o_pieces)
    end_t = time.time()
    print('{}'.format(end_t - start_t))

    start_t = time.time()
    for i in range(int(1e5)):
        board.check_for_winner_c(board.o_pieces)
    end_t = time.time()
    print('{}'.format(end_t - start_t))
    # print(sum(output)

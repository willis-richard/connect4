import numpy as np


class Connect4Stats():
    height = 6
    width = 7
    win_length = 4

    # Check inputs are sane
    assert width >= win_length and height >= win_length

    centre = (height / 2.0, width / 2.0)
    value_grid = np.stack([np.concatenate((np.arange(3.5), np.flip(np.arange(3)))) for i in range(height)])
    + np.transpose(np.stack([np.concatenate((np.arange(3), np.flip(np.arange(3)))) for i in range(width)]))
    # wtf centre not recognised as a variable...
    #value_grid = np.stack([np.concatenate((np.arange(centre[1]), np.flip(np.arange(int(centre[1]))))) for i in range(height)])
    #    + np.transpose(np.stack([np.concatenate((np.arange(centre[0]), np.flip(np.arange(int(centre[0]))))) for i in range(width)]))
    #value_grid = np.zeros((6,7))
    value_grid_t = np.transpose(value_grid)
    value_grid_sum = np.sum(value_grid)

    # my has function converts to a u64
    assert height * width <= 64

    hash_value = np.array([2**x for x in range(height * width)])

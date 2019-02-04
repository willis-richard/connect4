import numpy as np

from enum import Enum, IntEnum


class Connect4Stats():
    height = 6
    width = 7
    win_length = 4

    # Check inputs are sane
    assert width >= win_length and height >= win_length

    centre = (height / 2.0, width / 2.0)
    value_grid = np.stack([np.concatenate((np.arange(3.5),
                                           np.flip(np.arange(3))))
                           for i in range(height)]) \
        + np.transpose(
                     np.stack([np.concatenate((np.arange(3),
                                               np.flip(np.arange(3))))
                               for i in range(width)])) \
        + np.ones((height, width))
    # wtf centre not recognised as a variable...
    #value_grid = np.stack([np.concatenate((np.arange(centre[1]), np.flip(np.arange(int(centre[1]))))) for i in range(height)])
    #    + np.transpose(np.stack([np.concatenate((np.arange(centre[0]), np.flip(np.arange(int(centre[0]))))) for i in range(width)]))
    # value_grid = np.zeros((6,7))
    value_grid_t = np.transpose(value_grid)
    value_grid_sum = np.sum(value_grid)

    # my hash function converts to a u64
    assert height * width <= 64

    hash_value = np.array([2**x for x in range(height * width)])


class NetworkStats():
    channels = 2
    filters = 32
    n_fc_layers = 4
    n_residuals = 3
    area = Connect4Stats.height * Connect4Stats.width


class Side(IntEnum):
    o = 1
    x = 0


class Result(Enum):
    o_win = 1
    x_win = 0
    draw = 0.5


def same_side(result: Result, side: Side):
    if result == Result.o_win and side == Side.o:
        return True
    elif result == Result.x_win and side == Side.x:
        return True
    return False


def value_to_side(value: float, side: Side):
    return value if side == Side.o else (1.0 - value)


def result_to_side(result: Result, side: Side):
    return value_to_side(result.value, side)


def augment_data(board_train, value_train):
    v_wins = value_train[value_train == 1.0]
    v_draws = value_train[value_train == 0.5]
    v_losses = value_train[value_train == 0.0]
    b_wins = board_train[value_train == 1.0]
    b_draws = board_train[value_train == 0.5]
    b_losses = board_train[value_train == 0.0]

    n_wins = len(v_wins)
    n_draws = len(v_draws)
    n_losses = len(v_losses)

    v_augmented_draws = np.repeat(v_draws, n_wins/n_draws)
    v_augmented_losses = np.repeat(v_losses, n_wins/n_losses)
    b_augmented_draws = np.repeat(b_draws, n_wins/n_draws, axis=0)
    b_augmented_losses = np.repeat(b_losses, n_wins/n_losses, axis=0)

    extra_draw_idx = np.random.choice(range(len(v_draws)), n_wins - len(v_augmented_draws), replace=False)
    extra_losses_idx = np.random.choice(range(len(v_losses)), n_wins - len(v_augmented_losses), replace=False)

    v_augmented_draws = np.hstack([v_augmented_draws, v_draws[extra_draw_idx]])
    v_augmented_losses = np.hstack([v_augmented_losses, v_losses[extra_losses_idx]])
    b_augmented_draws = np.concatenate([b_augmented_draws, b_draws[extra_draw_idx]], axis=0)
    b_augmented_losses = np.concatenate([b_augmented_losses, b_losses[extra_losses_idx]], axis=0)

    value_train = np.hstack([v_wins, v_augmented_draws, v_augmented_losses])
    board_train = np.concatenate([b_wins, b_augmented_draws, b_augmented_losses], axis=0)

    return board_train, value_train

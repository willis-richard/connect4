from connect4 import board
from connect4 import player

import anytree
import pytest

import numpy as np


def test_next_move_winner():
    o_pieces = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 0],
         [1, 0, 0, 1, 0, 0, 0]], dtype=np.bool_)

    x_pieces = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 1, 0, 1, 0, 0]], dtype=np.bool_)

    board_ = board.Board(o_pieces=o_pieces,
                        x_pieces=x_pieces)

    computer = player.ComputerPlayer("test_name", 1, board_, 1)

    assert computer.make_move() == 1
    return


def test_stop_winner():
    o_pieces = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 0],
         [1, 0, 0, 1, 0, 0, 0]], dtype=np.bool_)

    x_pieces = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 0]], dtype=np.bool_)

    board_ = board.Board(o_pieces=o_pieces,
                        x_pieces=x_pieces)

    computer = player.ComputerPlayer("test_name", -1, board_, 2)

    board_.display()

    move = computer.make_move()

    for pre, fill, node in anytree.RenderTree(computer.tree.root):
        print("%s%s%s%s" % (pre, node.name, node.data.node_eval.tree_value, node.data.node_eval.evaluation))

    assert move == 1
    return

def test_stop_winner_2():
    o_pieces = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0]], dtype=np.bool_)

    x_pieces = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0]], dtype=np.bool_)

    board_ = board.Board(o_pieces=o_pieces,
                        x_pieces=x_pieces)
    board_.display()

    computer = player.ComputerPlayer("test_name", 1, board_, 2)

    move = computer.make_move()

    for pre, fill, node in anytree.RenderTree(computer.tree.root):
        print("%s%s%s%s" % (pre, node.name, node.data.node_eval.tree_value, node.data.node_eval.evaluation))

    assert move == 1

    return

def test_stop_winner_3():
#[[' ' ' ' ' ' 'x' 'x' ' ' ' ']
# [' ' ' ' ' ' 'x' 'x' 'x' ' ']
# [' ' ' ' ' ' 'x' 'o' 'o' ' ']
# [' ' ' ' ' ' 'o' 'x' 'x' 'o']
# [' ' ' ' 'o' 'x' 'o' 'o' 'o']
# [' ' 'x' 'o' 'x' 'o' 'o' 'o']]
    o_pieces = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0],
         [0, 0, 0, 1, 0, 0, 1],
         [0, 0, 1, 0, 1, 1, 1],
         [0, 0, 1, 0, 1, 1, 1]], dtype=np.bool_)

    x_pieces = np.array(
        [[0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 1, 0, 0, 0]], dtype=np.bool_)

    board_ = board.Board(o_pieces=o_pieces,
                        x_pieces=x_pieces)
    board_.display()

    computer = player.ComputerPlayer("test_name", -1, board_, 4)

    move = computer.make_move()

    for pre, fill, node in anytree.RenderTree(computer.tree.root):
        print("%s%s%s%s" % (pre, node.name, node.data.node_eval.tree_value, node.data.node_eval.evaluation))

    assert move == 6
    return

def test_stop_winner_4():
# [[' ' ' ' ' ' ' ' ' ' ' ' ' ']
#  [' ' ' ' ' ' ' ' ' ' ' ' ' ']
#  [' ' ' ' ' ' ' ' ' ' ' ' ' ']
#  [' ' ' ' ' ' ' ' ' ' ' ' ' ']
#  [' ' ' ' ' ' 'x' ' ' ' ' ' ']
#  [' ' ' ' ' ' 'o' 'o' ' ' ' ']]
    o_pieces = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0]], dtype=np.bool_)

    x_pieces = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]], dtype=np.bool_)

    board_ = board.Board(o_pieces=o_pieces,
                        x_pieces=x_pieces)
    board_.display()

    computer = player.ComputerPlayer("test_name", -1, board_, 4)

    move = computer.make_move()

    for pre, fill, node in anytree.RenderTree(computer.tree.root):
        print("%s%s%s%s" % (pre, node.name, node.data.node_eval.tree_value, node.data.node_eval.evaluation))

    assert move in [2,5]

    return

"""
def test_stop_winner_5():
#[[' ' ' ' ' ' 'x' ' ' ' ' ' ']
# [' ' ' ' ' ' 'x' ' ' ' ' ' ']
# [' ' ' ' 'x' 'x' ' ' ' ' ' ']
# [' ' 'o' 'o' 'o' ' ' ' ' ' ']
# [' ' 'o' 'o' 'x' ' ' ' ' ' ']
# ['o' 'x' 'o' 'o' 'x' ' ' ' ']]
    o_pieces = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 0],
         [1, 0, 1, 1, 0, 0, 0]], dtype=np.bool_)

    x_pieces = np.array(
        [[0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 1, 0, 0]], dtype=np.bool_)

    board_ = board.Board(o_pieces=o_pieces,
                        x_pieces=x_pieces)
    board_.display()

    computer = player.ComputerPlayer("test_name", -1, board_, 4)

    move = computer.make_move()

    for pre, fill, node in anytree.RenderTree(computer.tree.root):
        print("%s%s%s%s" % (pre, node.name, node.data.node_eval.tree_value, node.data.node_eval.evaluation))

    assert move not in [3,4]

    return
"""

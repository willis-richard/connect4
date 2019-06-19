from oinkoink.board import Board

from oinkoink.neural.storage import GameStorage
from oinkoink.neural.training_game import GameData

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from typing import List, Sequence


class Connect4Dataset(Dataset):
    def __init__(self,
                 boards: torch.FloatTensor,
                 values: torch.FloatTensor,
                 priors: torch.FloatTensor):
        self.boards = boards
        self.values = values
        self.priors = priors

    def save(self, filename):
        data = {}
        data['boards'] = self.boards
        data['values'] = self.values
        data['priors'] = self.priors

        torch.save(data, filename)

    @classmethod
    def load(cls, filename):
        data = torch.load(filename)
        return cls(data['boards'], data['values'], data['priors'])

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx: int):
        if self.priors is None:
            return (self.boards[idx],
                    self.values[idx])
        else:
            return (self.boards[idx],
                    self.values[idx],
                    self.priors[idx])


class TrainingDataStorage(GameStorage):
    def td_file_name(self, folder_path, gen):
        return "{}/{}/data.pth".format(folder_path, gen)

    def save(self,
             games: List[GameData],
             folder_path: str):
        super().save(games, folder_path)

        data = np.sum(list(map(lambda x: x.data, games)))
        board_t, value_t, prior_t = native_to_pytorch(data.boards,
                                                      data.values,
                                                      data.priors,
                                                      add_fliplr=True)

        dataset = Connect4Dataset(board_t, value_t, prior_t)
        dataset.save(folder_path + '/data.pth')

    def get_dataset(self,
                    base_path: str,
                    gen: int):
        n = min(20, int((gen + 1) / 2))
        file_names = [self.td_file_name(base_path, i)
                      for i in range(gen,
                                     gen - n,
                                     -1)]
        return ConcatDataset([Connect4Dataset.load(f)
                              for f in file_names])


def native_to_pytorch(boards: List[Board],
                      values: Sequence[float],
                      priors: List[Sequence[float]] = None,
                      to_move_channel: bool = True,
                      add_fliplr: bool = False):
    assert len(boards) == len(values)

    if add_fliplr:
        flip_boards = list(map(lambda x: x.create_fliplr(), boards))
        boards.extend(flip_boards)
        values = np.concatenate((values, values), axis=None)
    boards_t = torch.FloatTensor(list(map(
        lambda x: x.to_array(), boards)))

    if not to_move_channel:
        boards_t = boards_t[:, 1:]
    values_t = torch.FloatTensor(values)

    if priors is None:
        priors_t = None
    else:
        if add_fliplr:
            flip_priors = list(map(lambda x: np.flip(x), priors))
            priors.extend(flip_priors)
        assert len(boards_t) == len(priors)
        priors_t = torch.FloatTensor(priors)

    return boards_t, values_t, priors_t

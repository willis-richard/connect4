import os

from connect4.neural.config import (AlphaZeroConfig,
                                    ModelConfig,
                                    StorageConfig)

config = AlphaZeroConfig(storage_config=StorageConfig(
    save_dir='{}/Downloads/nn/new_dir15'.format(os.path.expanduser('~'))),
                         # game_processes=1,
                         # game_threads=1,
                         n_training_games=1200,
                         n_eval=500)

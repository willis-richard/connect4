from oinkoink.neural.config import (AlphaZeroConfig,
                                    ModelConfig,
                                    NetConfig,
                                    StorageConfig)

import os

config = AlphaZeroConfig(model_config=ModelConfig(
    net_config=NetConfig(filters=64, n_fc_layers=6, n_residuals=6)),
                         storage_config=StorageConfig(
                             save_dir='{}/Downloads/nn/new_dir15'.
                             format(os.path.expanduser('~'))),
                         # game_processes=1,
                         # game_threads=1,
                         n_training_games=1200,
                         n_eval=500)

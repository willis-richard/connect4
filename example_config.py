from connect4.neural.config import (AlphaZeroConfig,
                                    ModelConfig,
                                    StorageConfig)

config = AlphaZeroConfig(storage_config=StorageConfig(save_dir='/home/richard/Downloads/nn/new_dir8'),
                         # game_processes=2,
                         # game_threads=2,
                         # n_training_games=4,
                         n_eval=1)

from connect4.neural.config import (AlphaZeroConfig,
                                    ModelConfig,
                                    StorageConfig)

config = AlphaZeroConfig(model_config=ModelConfig(
                         n_training_epochs=10),
                         storage_config=StorageConfig(save_dir='/home/richard/Downloads/nn/new_dir4'),
                         # game_processes=2,
                         # game_threads=1,
                         # n_training_games=2,
                         use_pytorch=True,
                         n_eval=1)

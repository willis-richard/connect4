from connect4.neural.config import (AlphaZeroConfig,
                                    ModelConfig,
                                    StorageConfig)

config = AlphaZeroConfig(model_config=ModelConfig(
                         n_training_epochs=2),
                         storage_config=StorageConfig(save_dir='/home/richard/Downloads/nn/new_dir3'),
                         game_processes=2,
                         game_threads=2,
                         n_training_games=8,
                         use_pytorch=True,
                         n_eval=1)

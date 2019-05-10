from connect4.neural.config import (AlphaZeroConfig,
                                    ModelConfig,
                                    StorageConfig)

config = AlphaZeroConfig(model_config=ModelConfig(
                         n_training_epochs=2),
                         storage_config=StorageConfig(save_dir='/home/richard/Downloads/nn/new_dir3'),
                         game_processes=10,
                         game_threads=10,
                         n_training_games=1000,
                         use_pytorch=True,
                         n_eval=1)

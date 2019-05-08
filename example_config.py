from connect4.neural.config import AlphaZeroConfig, StorageConfig

config = AlphaZeroConfig(storage_config=StorageConfig(save_dir='/home/richard/Downloads/nn/new_dir3'),
                         game_processes=10,
                         game_threads=20,
                         n_training_epochs=2,
                         n_training_games=600,
                         use_pytorch=True,
                         n_eval=1)

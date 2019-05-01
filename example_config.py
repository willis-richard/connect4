from connect4.neural.config import AlphaZeroConfig, StorageConfig

# config = AlphaZeroConfig(simulations=100,
#                          n_training_games=10,
#                          n_training_epochs=2,
#                          use_pytorch=True)

config = AlphaZeroConfig(storage_config=StorageConfig(save_dir='/home/richard/Downloads/nn/new_dir2'),
                         agents=1, #5
                         n_training_epochs=1,
                         n_training_games=4, #1000
                         use_pytorch=True)

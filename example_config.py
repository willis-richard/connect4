from connect4.neural.config import AlphaZeroConfig

config = AlphaZeroConfig(simulations=100,
                         n_training_games=10,
                         n_training_epochs=2,
                         use_pytorch=True)

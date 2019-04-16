from connect4.neural.config import AlphaZeroConfig

config = AlphaZeroConfig(simulations=20,
                         n_training_games=2,
                         n_training_epochs=2,
                         use_pytorch=True)

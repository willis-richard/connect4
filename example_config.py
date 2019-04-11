from connect4.neural.config import AlphaZeroConfig

config = AlphaZeroConfig(simulations=200,
                         n_training_games=20,
                         n_training_epochs=2,
                         use_pytorch=True)

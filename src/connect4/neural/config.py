class ModelConfig():
    weight_decay = 1e-4
    momentum = 0.9
    # Schedule for chess and shogi, Go starts at 2e-2 immediately.
    initial_lr = 0.2
    milestones = [int(1e3), int(3e3), int(5e3)]
    gamma = 0.1
    # read https://pytorch.org/docs/stable/optim.html about pre layer lr
    # could have one for the body, and diff ones for each head


class AlphaZeroConfig():
    agents = 12
    batch_size = 1
    n_eval = 2
    n_training_epochs = 2
    n_training_games = 2
    window_size = 5000
    model_config = ModelConfig()

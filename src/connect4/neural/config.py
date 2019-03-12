
class ModelConfig():
    weight_decay = 1e-4
    momentum = 0.9
    # Schedule for chess and shogi, Go starts at 2e-2 immediately.
    initial_lr = 0.2
    milestones = [int(1e3), int(3e3), int(5e3)]
    gamma = 0.1
    # read https://pytorch.org/docs/stable/optim.html about pre layer lr
    # could have one for the body, and diff ones for each head
    batch_size = 1
    window_size = 50

class StorageConfig():
    path_8ply_boards = '/home/richard/Downloads/connect4_boards.pth'
    path_8ply_values = '/home/richard/Downloads/connect4_values.pth'
    save_dir = '/home/richard/Downloads/nn/new_dir'

class AlphaZeroConfig():
    agents = 1
    simulations = 100
    pb_c_init=9999
    n_eval = 2
    n_training_epochs = 2
    n_training_games = 2

    model_config = ModelConfig()
    storage_config = StorageConfig()

    vizdom_enabled = False

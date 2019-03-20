
class ModelConfig():
    def __init__(self,
                 weight_decay=1e-4,
                 momentum=0.9,
                 # Schedule for chess and shogi, Go starts at 2e-2 immediately.
                 initial_lr=0.05,
                 # Remember that these are in epochs, and we have a n_training_epochs parameter
                 milestones=[int(1e3), int(3e3), int(5e3)],
                 gamma=0.1,
                 # read https://pytorch.org/docs/stable/optim.html about pre layer lr
                 # could have one for the body, and diff ones for each head
                 batch_size=1):
         self.weight_decay = weight_decay
         self.momentum = momentum
         self.initial_lr = initial_lr
         self.milestones = milestones
         self.gamma = gamma
         self.batch_size = batch_size


class StorageConfig():
    def __init__(self,
                 path_8ply_boards='/home/richard/Downloads/connect4_boards.pth',
                 path_8ply_values='/home/richard/Downloads/connect4_values.pth',
                 save_dir='/home/richard/Downloads/nn/new_dir4'):
        self.path_8ply_boards = path_8ply_boards
        self.path_8ply_values = path_8ply_values
        self.save_dir = save_dir


class AlphaZeroConfig():
    def __init__(self,
                 model_config=ModelConfig(),
                 storage_config=StorageConfig(),
                 agents=1,
                 simulations=100,
                 pb_c_init=9999,
                 root_dirichlet_alpha=1.0, # 0.3 for chess, 0.03 for Go and 0.15 for shogi.
                 root_exploration_fraction=0.25,
                 n_eval=2,
                 n_training_epochs=2,
                 n_training_games=2,
                 visdom_enabled=False):
        self.model_config = model_config
        self.storage_config = storage_config
        self.agents = agents
        self.simulations = simulations
        self.pb_c_init = pb_c_init
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction
        self.n_eval = n_eval
        self.n_training_epochs = n_training_epochs
        self.n_training_games = n_training_games
        self.visdom_enabled = visdom_enabled



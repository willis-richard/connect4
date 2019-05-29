
class ModelConfig():
    def __init__(self,
                 weight_decay=1e-4,
                 momentum=0.9,
                 # Schedule for chess and shogi, Go starts at 2e-2 immediately.
                 initial_lr=0.005,
                 # Remember that these are in epochs, and we have a n_training_epochs parameter
                 milestones=[int(3e2), int(6e2), int(9e2)],
                 gamma=0.1,
                 # read https://pytorch.org/docs/stable/optim.html about pre layer lr
                 # could have one for the body, and diff ones for each head
                 batch_size=4096,
                 n_training_epochs=10,
                 use_gpu=True):
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.initial_lr = initial_lr
        self.milestones = milestones
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_training_epochs = n_training_epochs
        self.use_gpu = use_gpu


class StorageConfig():
    def __init__(self,
                 path_8ply_boards='/home/richard/data/connect4/connect4_boards.pth',
                 path_8ply_values='/home/richard/data/connect4/connect4_values.pth',
                 save_dir='/home/richard/Downloads/nn/new_dir'):
        self.path_8ply_boards = path_8ply_boards
        self.path_8ply_values = path_8ply_values
        self.save_dir = save_dir


class AlphaZeroConfig():
    def __init__(self,
                 model_config=ModelConfig(),
                 storage_config=StorageConfig(),
                 game_processes=10,
                 game_threads=20,
                 simulations=800,
                 pb_c_base=19652,
                 pb_c_init=1.25,
                 root_dirichlet_alpha=1.0, # 0.3 for chess, 0.03 for Go and 0.15 for shogi.
                 root_exploration_fraction=0.25,
                 num_sampling_moves=8,
                 n_eval=1,
                 n_training_games=1200,
                 use_pytorch=True,
                 enable_gpu=True,
                 visdom_enabled=False):
        self.model_config = model_config
        self.storage_config = storage_config
        self.game_processes = game_processes
        self.game_threads = game_threads
        self.simulations = simulations
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction
        self.num_sampling_moves = num_sampling_moves
        self.n_eval = n_eval
        self.n_training_games = n_training_games
        self.use_pytorch = use_pytorch
        self.visdom_enabled = visdom_enabled

import os


dirname = os.path.dirname(os.path.abspath(__file__))


class ModelConfig():
    def __init__(self,
                 weight_decay=1e-4,
                 momentum=0.9,
                 initial_lr=0.01,
                 # These are in training generations
                 milestones=[int(150), int(350), int(750)],
                 gamma=0.1,
                 batch_size=4096,
                 n_training_epochs=5,
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
                 save_dir=os.path.expanduser('~'),
                 data_dir='{}/../data'.format(dirname)):
        self.save_dir = save_dir
        self.data_dir = data_dir


class AlphaZeroConfig():
    def __init__(self,
                 model_config=ModelConfig(),
                 storage_config=StorageConfig(),
                 game_processes=10,
                 game_threads=20,
                 simulations=800,
                 pb_c_base=19652,
                 pb_c_init=1.25,
                 # 0.3 for chess, 0.03 for Go and 0.15 for shogi.
                 root_dirichlet_alpha=0.3,
                 root_exploration_fraction=0.25,
                 num_sampling_moves=6,
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

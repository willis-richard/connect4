import connect4.evaluators as ev
from connect4.match import Match
from connect4.mcts import MCTS, MCTSConfig

from connect4.neural.config import ModelConfig
from connect4.neural.nn_pytorch import ModelWrapper

from functools import partial
import sys
from torch.multiprocessing import set_start_method


if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError as e:
        if str(e) == 'context has already been set':
            pass
    for i in range(0, 250, 50):
        model_1 = ModelWrapper(ModelConfig(),
                             file_name=sys.argv[1] + str(i) + '.pth')

        a0_1 = MCTS("mcts_nn_" + str(i),
                    MCTSConfig(simulations=800),
                    ev.Evaluator(partial(ev.evaluate_nn,
                                         model=model_1)))

        for j in [250]:
            if j > i:
                model_2 = ModelWrapper(ModelConfig(),
                                     file_name=sys.argv[1] + str(j) + '.pth')

                a0_2 = MCTS("mcts_nn_" + str(j),
                            MCTSConfig(simulations=800),
                            ev.Evaluator(partial(ev.evaluate_nn,
                                                 model=model_2)))
                match = Match(False,
                              a0_1,
                              a0_2,
                              plies=2,
                              switch=True)
                match.play(5)

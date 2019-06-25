import oinkoink.evaluators as ev
from oinkoink.match import Match
from oinkoink.mcts import MCTS, MCTSConfig

from oinkoink.neural.config import ModelConfig, NetConfig
from oinkoink.neural.pytorch.model import ModelWrapper

from functools import partial
import pandas as pd
import sys
from torch.multiprocessing import set_start_method


if __name__ == "__main__":
    SIMULATIONS = 800
    try:
        set_start_method('spawn')
    except RuntimeError as e:
        if str(e) == 'context has already been set':
            pass

    df = pd.DataFrame()
    index = []

    for i in range(40, 121, 20):
        model_1 = ModelWrapper(ModelConfig(net_config=
            NetConfig(filters=64,
                      n_fc_layers=6,
                      n_residuals=6)),
                               file_name="{}/{}/net.pth".format(
                                   sys.argv[1], i))

        a0_1 = MCTS("mcts_nn_" + str(i),
                    MCTSConfig(simulations=SIMULATIONS),
                    ev.Evaluator(partial(ev.evaluate_nn,
                                         model=model_1)))

        for j in range(20, 101, 20):
            if j > i:
                model_2 = ModelWrapper(ModelConfig(net_config=
                    NetConfig(filters=64,
                              n_fc_layers=6,
                              n_residuals=6)),
                                       file_name="{}/{}/net.pth".format(
                                           sys.argv[1], j))

                a0_2 = MCTS("mcts_nn_" + str(j),
                            MCTSConfig(simulations=SIMULATIONS),
                            ev.Evaluator(partial(ev.evaluate_nn,
                                                 model=model_2)))
                match = Match(False,
                              a0_1,
                              a0_2,
                              plies=2,
                              switch=True)
                results = match.play(2)
                df = df.append(results, ignore_index=True)
                index.append('{} vs {}'.format(i, j))

    df['name'] = index
    df.set_index('name', inplace=True)
    print(df)
    df.to_pickle('{}/head_to_head_results.pkl'.format(sys.argv[1]))

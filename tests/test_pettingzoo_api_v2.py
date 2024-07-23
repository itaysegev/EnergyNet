import unittest
from pettingzoo.test import parallel_api_test

import sys
import os

# Add the project's root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from energy_net.env.energy_net_v0 import parallel_env
from tests.single_agent_config import test_env_cfgs

NUM_CYCLES = 1000
NUM_SEEDS = 5
INITIAL_SEED = 42


class TestParallelAPI(unittest.TestCase):
    def test_parallel_api(self):
        for env_name, env_cfg in test_env_cfgs.items():
            self.__multi_seed_api_test(parallel_env, env_name, env_cfg, parallel_api_test)

    def __multi_seed_api_test(self, env_init, name, cfg, pz_test):
        seed = hash(name)
        for _ in range(NUM_SEEDS):
            seed = abs(hash(str(seed)))
            print("Testing seed: ", seed)
            pz_test(env_init(**cfg, initial_seed=seed), num_cycles=NUM_CYCLES)


if __name__ == '__main__':
    unittest.main()
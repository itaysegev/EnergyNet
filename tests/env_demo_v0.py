import warnings
from typing import Mapping, List, Union, Any

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the project's root directory to sys.path
from energy_net.env.single_entity_v0 import gym_env
from energy_net.model.reward import RewardFunction
from energy_net.model.action import StorageAction
from tests.single_agent_config import single_agent_cfgs


from stable_baselines3 import SAC
import matplotlib.pyplot as plt
import numpy as np

class pcsunitDummyRewardFunction(RewardFunction):
    r"""Dummy reward function class.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    **kwargs : dict
        Other keyword arguments for custom reward calculation.
    """

    def __init__(self, env_metadata: Mapping[str, Any], **kwargs):
        super().__init__(env_metadata, **kwargs)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        r"""Calculates reward.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """
        # observation_seperator = importlib.import_module('utils.env_utils').observation_seperator
        return sum(d.consumption for d in observation_seperator(observations) if isinstance(d, ConsumerState))


def test_gym_api():
    rewards = []
    actions = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for env_name, env_cfg in single_agent_cfgs.items():
            seed = hash(env_name)
            seed = abs(hash(str(seed)))
            env = gym_env(**env_cfg, initial_seed=seed)
            model = SAC("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=100)
            vec_env = model.get_env()
            obs = vec_env.reset()
            print(obs, "reset")
            # observation, info = env.reset()
            for _ in range(4):
                # action = StorageAction(charge=env.action_space.sample().item())  # agents policy that uses the observation and info
                # action, _ = model.predict(obs, deterministic=True)
                action = env.action_space.sample()
                print(action, "action")
                observation, reward, terminated, truncated, info = env.step(action)
                
                rewards.append(reward)
                actions.append(action.item())
                
                
                print(observation, "obs")
                
                if terminated or truncated:
                    observation, info = env.reset()
                    
        env.close()
        
    

if __name__ == '__main__':
    test_gym_api()
    


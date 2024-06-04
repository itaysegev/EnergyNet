import warnings
from typing import Mapping, List, Union, Any

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the project's root directory to sys.path
from energy_net.env.single_entity_v0 import gym_env
from energy_net.elinor_simulation.common import simulator_cfgs


def simulator():
    rewards = []
    actions = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for env_name, env_cfg in simulator_cfgs.items():
            seed = hash(env_name)
            seed = abs(hash(str(seed)))
            env = gym_env(**env_cfg, initial_seed=seed)
            
            observation, _ = env.reset()
            print(observation, "obs")
            print("#####################")


            for _ in range(1_000):
                # action = StorageAction(charge=env.action_space.sample().item())  # agents policy that uses the observation and info
                # action, _ = model.predict(obs, deterministic=True)
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                print(observation[1], "obs")
                
                
                rewards.append(reward)
                actions.append(action.item())
                
                
                
                if terminated or truncated:
                    observation, info = env.reset()
                    
        env.close()
        
    

if __name__ == '__main__':
    simulator()
    


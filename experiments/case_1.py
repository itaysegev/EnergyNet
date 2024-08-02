from energy_net.env.single_entity_v0 import gym_env
from tests.single_agent_config import single_agent_cfgs

from energy_net.env.EnergyNetEnv import EnergyNetEnv
from tests.test_network import default_pcsunit
from energy_net.network import Network
from energy_net.stratigic_entity import StrategicEntity

from stable_baselines3.common.env_checker import check_env
from train import train
import time
import os




ALGO = ['ppo', 'sac', 'a2c']

def simulation_reward_function(state, action, new_state):
    alpha = 0.01
    grid_electricity = action.item() + state.get_consumption() - state.get_production()
    
    if grid_electricity < 0:
        return -1 *1000
    price = grid_electricity
    return -1 * price * grid_electricity 



def losses_simulation_reward_function(state, action, new_state):
    alpha = 0.01
    grid_electricity = action.item() + state.get_consumption() - state.get_production()
    price = grid_electricity
    price = price + alpha * price * price
    return -1 * price * price

def main():

    strategic_entities = [StrategicEntity(name="pcs_agent", network_entity=default_pcsunit(), reward_function=losses_simulation_reward_function)]
    network = Network(name="test_network", strategic_entities=strategic_entities)

    env = gym_env(network=network, simulation_start_time_step=0,
                       simulation_end_time_step=48, episode_time_steps=48,
                       seconds_per_time_step=60*30, initial_seed=0)
    
    try:
        check_env(env)
        print('Passed test!! EnergyNetEnv is compatible with SB3 when using the StableBaselines3Wrapper.')
    finally:
        pass
    
    
    
    
    for algo in ALGO:
        model = train(env = env, algo=algo, tensorboard_log="./tmp/stable-baselines_case3/", trained_agent="", truncate_last_trajectory=True, n_timesteps=-1,
              num_threads=-1, log_interval=-1, eval_freq=10_000, optimization_log_path=None, eval_episodes=10, n_eval_envs=1, save_freq=10_000,
              save_replay_buffer=False, log_folder="case3_logs", seed=-1, vec_env="dummy", device="auto", n_trials=500, max_total_trials=None,
              optimize_hyperparameters=False, no_optim_plots=False, n_jobs=1, sampler="tpe", pruner="median", n_startup_trials=10)
        
    
if __name__ == '__main__':
    main()
    
    

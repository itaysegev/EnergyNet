from energy_net.env.single_entity_v0 import gym_env
from tests.single_agent_config import single_agent_cfgs

from energy_net.env.EnergyNetEnv import EnergyNetEnv
from tests.test_network import default_pcsunit, default_reward_function
from energy_net.network import Network
from energy_net.stratigic_entity import StrategicEntity

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import time
import os





def simulation_reward_function(state, action, new_state):
    grid_electricity = max(action.item() - state.get_consumption() + state.get_production(), 0)
    price = grid_electricity
    return -1 * price * grid_electricity


def main():
    
    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}/"
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
	    

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # simulate the environment
    strategic_entities = [StrategicEntity(name="pcs_agent", network_entity=default_pcsunit(), reward_function=simulation_reward_function)]
    network =  Network(name="test_network", strategic_entities=strategic_entities)
    
    env = gym_env(network=network, simulation_start_time_step=0,
                       simulation_end_time_step=48*900, episode_time_steps=48,
                       seconds_per_time_step=60*30, initial_seed=0)
    try:
        check_env(env)
        print('Passed test!! EnergyNetEnv is compatible with SB3 when using the StableBaselines3Wrapper.')
    finally:
        pass
    # model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
    # model.learn(total_timesteps=env.time_steps*3, log_interval=1)
    obs, _ = env.reset()
    print(obs, 'init_obs')
    TIMESTEPS = 10000
    iters = 0
    while True:
        # action, _ = model.predict(obs, deterministic=True)
        action = env.action_space.sample()
        obs, rewards, terms, truncs, infos = env.step(action)
        print(rewards, 'rewards')
        print(obs, 'obs')
        if terms or truncs:
            print('Episode ended')
            print(iters)
            break
        
        iters += 1
        
    env.close()
        
    

if __name__ == '__main__':
    main()
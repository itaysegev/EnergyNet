
import warnings

import numpy as np

from energy_net.agents.SACAgent import SACAgent
from energy_net.devices.params import StorageParams, ConsumptionParams, ProductionParams
from energy_net.env.EnergyNetEnv import EnergyNetEnv
from energy_net.model.action import EnergyAction, ProduceAction, StorageAction, ConsumeAction
from energy_net.dynamics.consumption_dynamics.consumption_dynamics import PCSUnitConsumptionDynamics
from energy_net.dynamics.production_dynamics.production_dynamics import PVDynamics
from energy_net.dynamics.storage_dynamics.storage_dynamics import BatteryDynamics
from energy_net.entities.pcsunit import PCSUnit
from energy_net.devices.params import StorageParams, ConsumptionParams, ProductionParams
from energy_net.config import DEFAULT_LIFETIME_CONSTANT
# Add the project's root directory to sys.path
from energy_net.env.single_entity_v0 import gym_env
from common import single_agent_cfgs, get_env_cfgs

def test_pcsunit():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # initialize consumer devices
        consumption_params_arr=[]
        consumption_params = ConsumptionParams(name='pcsunit_consumption', energy_dynamics=PCSUnitConsumptionDynamics(), lifetime_constant=DEFAULT_LIFETIME_CONSTANT)
        consumption_params_arr.append(consumption_params)
        consumption_params_dict = {'pcsunit_consumption': consumption_params}

        # initialize storage devices
        storage_params_arr=[]
        storage_params = StorageParams(name = 'test_battery', energy_capacity = 100, power_capacity = 200,inital_charge = 50, charging_efficiency = 1,discharging_efficiency = 1, lifetime_constant = 15, energy_dynamics = BatteryDynamics())
        storage_params_arr.append(storage_params)
        storage_params_dict = {'test_battery': storage_params}

        # initialize production devices
        production_params_arr=[]
        production_params = ProductionParams(name='test_pv', max_production=100, efficiency=0.9, energy_dynamics=PVDynamics())
        production_params_arr.append(production_params)
        production_params_dict = {'test_pv': production_params}

        # initilaize pcsunit
        pcsunit = PCSUnit(name="test_pcsunit", consumption_params_dict=consumption_params_dict, storage_params_dict=storage_params_dict, production_params_dict=production_params_dict, agg_func= None)

        # check individual actions
        pcsunit.perform_joint_action({'test_battery':StorageAction(charge=10), 'pcsunit_consumption': ConsumeAction(consume=None), 'test_pv': ProduceAction(produce=None)})


        # create wrapper for the environment
        #env = EnergyNetEnv(network_entities=[pcsunit],
        #                    simulation_start_time_step=0,
        #                    simulation_end_time_step=100,
        #                    episode_time_steps=100,
        #                    seconds_per_time_step=1,
        #                    reward_function=None,
        #                    random_seed=None)

        for env_name, env_cfg in single_agent_cfgs.items():
            seed = hash(env_name)
            seed = abs(hash(str(seed)))
            env_cfg['network_entities'] = [pcsunit]
            env = gym_env(**env_cfg, initial_seed=seed)

        # initialize agent
        agent = SACAgent(env, 'MlpPolicy', verbose=0, log_dir="/tmp/gym/")

        # train agent
        #agent.train(total_timesteps=100)

        # full train and evaluation process
        rewards, actions, soc = [], [], []
        eval_timesteps = 50
        train_timesteps = 100
        with warnings.catch_warnings():
                agent.train(total_timesteps=train_timesteps)
                observation = agent.env.reset()
                for _ in range(eval_timesteps):
                    soc.append(observation.squeeze()[0])
                    action = agent.get_action(observation, deterministic=True)
                    actions.append(action.item())
                    observation, reward, done, info = agent.env.step(action)
                    rewards.append(reward.item())

                    if done:
                        observation, info = env.reset()
        env.close()

    '''
    # run simulation
    for _ in range(1000):
        action = env.action_space.sample()  # agents policy that uses the observation and info
        print("action:", action)
        observation, reward, terminated, truncated, info = env.step(action)
        #print(observation, "obs")

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    '''


if __name__ == '__main__':
    test_pcsunit()


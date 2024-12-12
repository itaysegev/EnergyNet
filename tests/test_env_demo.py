from stable_baselines3.common.env_checker import check_env
import time
from energy_net.env.single_entity_v0 import gym_env
from stable_baselines3 import PPO
from tests.test_network import default_pcsunit
from energy_net.network import Network
from energy_net.components.pcsunit import PCSUnit
from energy_net.components.params import StorageParams, ProductionParams, ConsumptionParams, DeviceParams
from energy_net.dynamics.consumption_dynamics.consumption_dynamics import GeneralLoad
from energy_net.dynamics.production_dynamics.production_dynamics import PVDynamics
from energy_net.dynamics.storage_dynamics.storage_dynamics import BatteryDynamics
from energy_net.dynamics.grid_dynamics import GridDynamics

from energy_net.components.grid_device import GridDevice
from  energy_net.components.storage_devices.local_storage import Battery
from  energy_net.components.consumption_devices.local_consumer import ConsumerDevice
from  energy_net.components.production_devices.local_producer import PrivateProducer
from energy_net.config import DEFAULT_LIFETIME_CONSTANT
from energy_net.stratigic_entity import StrategicEntity
import os
import pandas as pd



def test_pcsunit():
    # initialize consumer components
        consumption_params_arr=[]
        file_name = 'first_day_data.xlsx'
        value_row_name = 'El [MWh]'
        time_row_name = 'Hour'
    
        general_load = GeneralLoad(file_name, value_row_name, time_row_name)
        consumption_params = ConsumptionParams(name='pcsunit_consumption', energy_dynamics=general_load, lifetime_constant=1, max_electric_power=general_load.max_electric_power)
        consumption_params_arr.append(consumption_params)
        consumption_params_dict = {'pcsunit_consumption': consumption_params}
        
        # initialize storage components
        storage_params_arr=[]
        storage_params = StorageParams(name = 'test_battery', energy_capacity = 4, power_capacity = 4,initial_charge = 0, charging_efficiency = 1,discharging_efficiency = 1, lifetime_constant = 1, energy_dynamics = BatteryDynamics())
        storage_params_arr.append(storage_params)
        storage_params_dict = {'test_battery': storage_params}

        # initialize production components
        production_params_arr=[]
        value_row_name = 'Epv [MWh]'

        
        pv_dynamics = PVDynamics(file_name, value_row_name, time_row_name)

        production_params = ProductionParams(name='test_pv', max_production=pv_dynamics.max_production, efficiency=1, energy_dynamics=pv_dynamics)
        production_params_arr.append(production_params)
        production_params_dict = {'test_pv': production_params}
        
        
        
        grid_params = DeviceParams(name='grid', energy_dynamics=GridDynamics(), lifetime_constant=DEFAULT_LIFETIME_CONSTANT)
        grid = GridDevice(grid_params)
        sub_entities= {name: ConsumerDevice(params) for name, params in consumption_params_dict.items()}
        sub_entities.update({name: Battery(params) for name, params in storage_params_dict.items()})
        sub_entities.update({name: PrivateProducer(params) for name, params in production_params_dict.items()})
        sub_entities.update({grid.name: grid})
        # initilaize pcsunit
        return PCSUnit(name="test_pcsuint", sub_entities=sub_entities, agg_func= None)


def simulation_reward_function(state, action, new_state):
    alpha = 0.01
    grid_electricity = action.item() + state.get_consumption() - state.get_production()
    
    if grid_electricity < 0:
        return -1_000
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
    strategic_entities = [StrategicEntity(name="pcs_agent", grid_entity=default_pcsunit(), reward_function=simulation_reward_function)]
    network =  Network(name="test_network", strategic_entities=strategic_entities)
    
    env = gym_env(network=network, simulation_start_time_step=0,
                       simulation_end_time_step=48, episode_time_steps=48,
                       seconds_per_time_step=60*30, initial_seed=0)
    try:
        check_env(env)
        print('Passed test!! EnergyNetEnv is compatible with SB3 when using the StableBaselines3Wrapper.')
    finally:
        pass
    
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
    # model.learn(total_timesteps=1e6, log_interval=1, progress_bar=True)
    
    
    # simulate the environment
    strategic_entities = [StrategicEntity(name="pcs_agent", grid_entity=test_pcsunit(), reward_function=simulation_reward_function)]
    network =  Network(name="test_network", strategic_entities=strategic_entities)
    
    env = gym_env(network=network, simulation_start_time_step=0,
                       simulation_end_time_step=48, episode_time_steps=48,
                       seconds_per_time_step=60*30, initial_seed=0)
    
    
    # Reset the environment to get the initial observation
    obs, info = env.reset()

    # List to store actions
    actions = []
    rewards = []

    # Perform actions using the trained model for 48 time steps
    for timestep in range(48):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs)
        actions.append(action)
        rewards.append(reward)
        if terminated or truncated:
            obs, info = env.reset()

    # Create a DataFrame with the actions
    df_actions = pd.DataFrame(actions, columns=['Action'])
    df_rewards = pd.DataFrame(rewards,columns=['Reward'])

    # Save the DataFrame to a file
    print(df_rewards)
    output_file_path = 'actions.csv'
    df_actions.to_csv(output_file_path, index=False)

    print(f"Actions saved to {output_file_path}")

    env.close()
        
    

if __name__ == '__main__':
    main()
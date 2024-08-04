import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.env_util import make_vec_env

from tests.test_network import default_pcsunit, default_reward_function
from energy_net.network import Network
from energy_net.stratigic_entity import StrategicEntity
from energy_net.env.single_entity_v0 import gym_env

from energy_net.entities.pcsunit import PCSUnit
from energy_net.devices.params import StorageParams, ProductionParams, ConsumptionParams, DeviceParams
from energy_net.dynamics.consumption_dynamics.consumption_dynamics import GeneralLoad
from energy_net.dynamics.production_dynamics.production_dynamics import PVDynamics
from energy_net.dynamics.storage_dynamics.storage_dynamics import BatteryDynamics
from energy_net.dynamics.grid_dynamics import GridDynamics

from energy_net.devices.grid_device import GridDevice
from  energy_net.devices.storage_devices.local_storage import Battery
from  energy_net.devices.consumption_devices.local_consumer import ConsumerDevice
from  energy_net.devices.production_devices.local_producer import PrivateProducer


def test_pcsunit():
    # initialize consumer devices
        consumption_params_arr=[]
        file_name = 'first_day_data.xlsx'
        value_row_name = 'El [MWh]'
        time_row_name = 'Hour'
    
        general_load = GeneralLoad(file_name, value_row_name, time_row_name)
        consumption_params = ConsumptionParams(name='pcsunit_consumption', energy_dynamics=general_load, lifetime_constant=1, max_electric_power=general_load.max_electric_power, init_consum=general_load.init_power)
        consumption_params_arr.append(consumption_params)
        consumption_params_dict = {'pcsunit_consumption': consumption_params}
        
        # initialize storage devices
        storage_params_arr=[]
        storage_params = StorageParams(name = 'test_battery', energy_capacity = 4, power_capacity = 4,initial_charge = 0, charging_efficiency = 1,discharging_efficiency = 1, lifetime_constant = 1, energy_dynamics = BatteryDynamics())
        storage_params_arr.append(storage_params)
        storage_params_dict = {'test_battery': storage_params}

        # initialize production devices
        production_params_arr=[]
        value_row_name = 'Epv [MWh]'

        
        pv_dynamics = PVDynamics(file_name, value_row_name, time_row_name)

        production_params = ProductionParams(name='test_pv', max_production=pv_dynamics.max_production, efficiency=1, energy_dynamics=pv_dynamics, init_production = pv_dynamics.init_production)
        production_params_arr.append(production_params)
        production_params_dict = {'test_pv': production_params}
        
        
        
        grid_params = DeviceParams(name='grid', energy_dynamics=GridDynamics(), lifetime_constant=1)
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

# Define the function to train and save models
def train_and_save_models(env, timesteps=1e6):
    # Train PPO
    model_ppo = PPO('MlpPolicy', env, verbose=1)
    model_ppo.learn(total_timesteps=timesteps)
    model_ppo.save("ppo_model")

    # Train SAC
    model_sac = SAC('MlpPolicy', env, verbose=1)
    model_sac.learn(total_timesteps=timesteps)
    model_sac.save("sac_model")

    # Train A2C
    model_a2c = A2C('MlpPolicy', env, verbose=1)
    model_a2c.learn(total_timesteps=timesteps)
    model_a2c.save("a2c_model")

# Define the function to run models on a new environment and save results
def collect_observations(env, model, num_steps=48):
    
    # Initialize lists to store results
    observations = []
   
    # Run models on the new environment
    obs, _ = env.reset()
    for step in range(num_steps):
        hour = obs[1]
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs)
        
        if terminated or truncated:
            obs, _ = env.reset()
            
    return observations



# Define your environment

# simulate the environment

train_network =  Network(name="train_network", strategic_entities=[StrategicEntity(name="pcs_agent", network_entity=default_pcsunit(), reward_function=simulation_reward_function)])
env = gym_env(network=train_network, simulation_start_time_step=0,
                       simulation_end_time_step=48, episode_time_steps=48,
                       seconds_per_time_step=60*30, initial_seed=0)

# simulate the environment

test_network =  Network(name="test_network", strategic_entities=[StrategicEntity(name="pcs_agent", network_entity=test_pcsunit(), reward_function=simulation_reward_function)])
    
test_env = gym_env(network=test_network, simulation_start_time_step=0,
                       simulation_end_time_step=48, episode_time_steps=48,
                       seconds_per_time_step=60*30, initial_seed=0)




# Train and save models
train_and_save_models(env)

model_ppo = PPO.load("ppo_model")
model_sac = SAC.load("sac_model")
model_a2c = A2C.load("a2c_model")

# Run models on a new environment and save results

ppo_observations = collect_observations(test_env, model_ppo)
sac_observations = collect_observations(test_env, model_sac)
a2c_observations = collect_observations(test_env, model_a2c)

# Extract the required columns from observations
data = {
    'Hour': [obs[1] for obs in ppo_observations],   # Assuming obs[1] is the Hour
    'Load': [obs[3] for obs in ppo_observations],   # Assuming obs[2] is the Load
    'PV': [obs[10] for obs in ppo_observations],     # Assuming obs[3] is the PV
    'PPO soc': [obs[4] for obs in ppo_observations],  # Assuming obs[4] is the soc
    'SAC soc': [obs[4] for obs in sac_observations],  # Assuming obs[4] is the soc
    'A2C soc': [obs[4] for obs in a2c_observations]   # Assuming obs[4] is the soc
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to an Excel file
output_file_path = 'model_comparisons_case1.xlsx'
df.to_excel(output_file_path, index=False)

print(f"Results saved to {output_file_path}")
import pandas as pd
import os
from stable_baselines3 import PPO, SAC,  TD3

from energy_net.network import Network
from energy_net.stratigic_entity import StrategicEntity
from energy_net.env.single_entity_v0 import gym_env

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


from stable_baselines3.common.callbacks import BaseCallback

class ActionMaskCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ActionMaskCallback, self).__init__(verbose)
        
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass
    
    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        self.model.action_space = self.model.env.envs[0].action_space
        self.training_env.action_space = self.model.env.envs[0].action_space
        # Apply mask to the action space in the model
        self.model.policy.action_space = self.model.action_space
        return True
    
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass
    
    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass



def build_pcsunit(file_name, value_row_name='El [MWh]', time_row_name= 'Hour', efficiency=1):
    # initialize consumer components
        consumption_params_arr=[]
        # file_name = 'first_day_data.xlsx'
        
    
        general_load = GeneralLoad(file_name, value_row_name, time_row_name)
        consumption_params = ConsumptionParams(name='pcsunit_consumption', energy_dynamics=general_load, lifetime_constant=1, max_electric_power=general_load.max_electric_power, init_consum=general_load.init_power)
        consumption_params_arr.append(consumption_params)
        consumption_params_dict = {'pcsunit_consumption': consumption_params}
        
        # initialize storage components
        storage_params_arr=[]
        storage_params = StorageParams(name = 'test_battery', energy_capacity = 4, power_capacity = 4, initial_charge = 0, charging_efficiency = efficiency, discharging_efficiency = efficiency, lifetime_constant = 1, energy_dynamics = BatteryDynamics())
        storage_params_arr.append(storage_params)
        storage_params_dict = {'test_battery': storage_params}

        # initialize production components
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

# Define the function to train and save models
def train_and_save_models(env, path, timesteps=1e5):
     # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    tensoerboard_log = os.path.join(path, "tensorboard")
    
    # Train PPO
    model_ppo = PPO('MlpPolicy', env, verbose=1, n_steps=48, tensorboard_log=tensoerboard_log)
    model_ppo.learn(total_timesteps=timesteps, callback=ActionMaskCallback(), progress_bar=True)
    model_ppo.save(os.path.join(path, "ppo_model"))
    
    
    
    # Train SAC
    model_sac = SAC('MlpPolicy', env, verbose=1, tensorboard_log=tensoerboard_log)
    model_sac.learn(total_timesteps=timesteps, callback=ActionMaskCallback(), progress_bar=True)
    model_sac.save(os.path.join(path, "sac_model"))
    
    

    # Train TD3
    model_td3 = TD3('MlpPolicy', env, verbose=1, tensorboard_log=tensoerboard_log)
    model_td3.learn(total_timesteps=timesteps, callback=ActionMaskCallback(), progress_bar=True )
    model_td3.save(os.path.join(path, "td3_model"))

# Define the function to run models on a new environment and save results
def collect_observations(env, model, num_steps=48):
    
    # Initialize lists to store results
    observations = []
    
    
   
    # Run models on the new environment
    
    obs, _ = env.reset()
    
    for step in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        # model.action_space = env.action_space
        # model.policy.action_space = env.action_space
        observations.append(obs)
        
        if term or trunc:
            obs, _ = env.reset()
            
    return observations


def create_gym_env(name, network_entity, reward_function, start_time_step=0, end_time_step=48, episode_time_steps=48, seconds_per_time_step=60*30, initial_seed=0):
    network = Network(name=name, strategic_entities=[StrategicEntity(name="pcs_agent", grid_entity=network_entity, reward_function=reward_function)])
    env = gym_env(network=network, simulation_start_time_step=start_time_step, simulation_end_time_step=end_time_step, episode_time_steps=episode_time_steps, seconds_per_time_step=seconds_per_time_step, initial_seed=initial_seed)
    return env

def save_observations_to_excel(ppo_observations, sac_observations, td3_observations, output_file_path):
    # Extract the required columns from observations
    data = {
        'Hour': [obs[1] for obs in ppo_observations],   # Assuming obs[1] is the Hour
        'Load': [obs[3] for obs in ppo_observations],   # Assuming obs[3] is the Load
        'PV': [obs[10] for obs in ppo_observations],    # Assuming obs[10] is the PV
        'PPO soc': [obs[4] for obs in ppo_observations], # Assuming obs[4] is the soc
        'SAC soc': [obs[4] for obs in sac_observations], # Assuming obs[4] is the soc
        'TD3 soc': [obs[4] for obs in td3_observations]  # Assuming obs[4] is the soc
    }

    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Ensure the output file path ends with '.xlsx'
    if not output_file_path.endswith('.xlsx'):
        output_file_path += '.xlsx'
        
    # Create directory if it doesn't exist
    directory = os.path.dirname(output_file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Save to an Excel file
    df.to_excel(output_file_path, index=False)

    print(f"Results saved to {output_file_path}")




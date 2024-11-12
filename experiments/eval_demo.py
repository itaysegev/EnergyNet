
from energy_net.env.single_entity_v0 import gym_env
from stable_baselines3 import PPO
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
from stable_baselines3.common.env_checker import check_env

def simulation_reward_function(state, action, new_state):
    alpha = 0.01
    grid_electricity = max(action.item() - state.get_consumption() + state.get_production(), 0)
    price = grid_electricity
    return -1 * price * grid_electricity


def test_pcsunit():
    # initialize consumer components
        consumption_params_arr=[]
        file_name = 'first_day_data.xlsx'
        value_row_name = 'El [MWh]'
        time_row_name = 'Hour'
    
        general_load = GeneralLoad(file_name, value_row_name, time_row_name)
        consumption_params = ConsumptionParams(name='pcsunit_consumption', energy_dynamics=general_load, lifetime_constant=DEFAULT_LIFETIME_CONSTANT, max_electric_power=general_load.max_electric_power)
        consumption_params_arr.append(consumption_params)
        consumption_params_dict = {'pcsunit_consumption': consumption_params}
        
        # initialize storage components
        storage_params_arr=[]
        storage_params = StorageParams(name = 'test_battery', energy_capacity = 4*1e7, power_capacity = 4*1e7,initial_charge = 0, charging_efficiency = 0.9,discharging_efficiency = 0.9, lifetime_constant = 1, energy_dynamics = BatteryDynamics())
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


# Example usage
if __name__ == "__main__":
    
    strategic_entities = [StrategicEntity(name="pcs_agent", network_entity=test_pcsunit(), reward_function=simulation_reward_function)]
    network = Network(name="test_network", strategic_entities=strategic_entities)

    env = gym_env(network=network, simulation_start_time_step=0,
                       simulation_end_time_step=48, episode_time_steps=48,
                       seconds_per_time_step=60*30)
    
    try:
        check_env(env)
        print('Passed test!! EnergyNetEnv is compatible with SB3 when using the StableBaselines3Wrapper.')
    finally:
        pass
    
    
    # Define the path to the model file
    model_path = "case1_logs/ppo/energy_net-v0_1/best_model.zip"
    
    model = PPO.load(model_path)
    
    
    # Reset the environment to get the initial observation
    obs, _ = env.reset()
    

    # Perform actions using the trained model
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, terms, truncs, infos = env.step(action)
        print(rewards, 'rewards')
        print(obs, 'obs')
        if terms or truncs:
            print('Episode ended')
            break

    env.close()

import unittest
import numpy as np

from energy_net.network import Network

from energy_net.entities.network_entity import CompositeNetworkEntity
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
from energy_net.config import DEFAULT_LIFETIME_CONSTANT
from energy_net.stratigic_entity import StrategicEntity

from energy_net.model.action import StorageAction, ProduceAction, ConsumeAction


from stable_baselines3 import PPO

def default_pcsunit():
    # initialize consumer devices
        consumption_params_arr=[]
        file_name = 'train_data.xlsx'
        value_row_name = 'El [MWh]'
        time_row_name = 'Hour'
    
        general_load = GeneralLoad(file_name, value_row_name, time_row_name)
        consumption_params = ConsumptionParams(name='pcsunit_consumption', energy_dynamics=general_load, lifetime_constant=1, max_electric_power=general_load.max_electric_power, init_consum=general_load.init_power)
        consumption_params_arr.append(consumption_params)
        consumption_params_arr.append(consumption_params)
        consumption_params_dict = {'pcsunit_consumption': consumption_params}
        
        # initialize storage devices
        storage_params_arr=[]
        storage_params = StorageParams(name = 'test_battery', energy_capacity = 4, power_capacity = 4, initial_charge = 0, charging_efficiency = 1,discharging_efficiency = 1, lifetime_constant = 1, energy_dynamics = BatteryDynamics())
        storage_params_arr.append(storage_params)
        storage_params_dict = {'test_battery': storage_params}

        # initialize production devices
        production_params_arr=[]
        value_row_name = 'Epv [MWh]'

        
        pv_dynamics = PVDynamics(file_name, value_row_name, time_row_name)

        production_params = ProductionParams(name='test_pv', max_production=pv_dynamics.max_production, efficiency=1, energy_dynamics=pv_dynamics, init_production = pv_dynamics.init_production)
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

def default_reward_function(state, action, new_state):
    return 0

def default_network():
    
    strategic_entities = [StrategicEntity(name="pcs_agent", network_entity=default_pcsunit(), reward_function=default_reward_function)]
    return Network(name="test_network", strategic_entities=strategic_entities)


# Write unit tests to test the initialization of the network and the state of the network
class TestNetwork(unittest.TestCase):
    def setUp(self):
        # Assuming `default_network` is a function that returns a default Network instance for testing
        self.network = default_network()

    def test_initialization(self):
        # Test the initialization of your Network
        self.assertEqual(self.network.name, "test_network")
        
        # Test the initialization of the Network's strategic entities
        self.assertEqual(self.network.strategic_entities['pcs_agent'].name, "pcs_agent")
        self.assertEqual(self.network.strategic_entities['pcs_agent'].network_entity.name, "test_pcsuint")
        self.assertEqual(self.network.strategic_entities['pcs_agent'].reward_function, default_reward_function)
        
        

    def test_observation_space(self):
        # Test the get_observation_space method of your Network
        obs_space = self.network.get_observation_space()
        self.assertEqual(len(obs_space['pcs_agent']), 4)
        
        
        
    def test_action_space(self):
        # Test the get_action_space method of your Network
        action_space = self.network.get_action_space()
        self.assertEqual(len(action_space), 1)
        
        
    def test_state(self):
        # Test the get_state method of your Network
        state = self.network.get_state()
        self.network.step({'pcs_agent': np.array([1.0])})
        state1 = self.network.get_state()
        self.network.step({'pcs_agent': np.array([1.0])})
        state2 = self.network.get_state()
        self.network.step({'pcs_agent': np.array([1.0])})
        state3 = self.network.get_state()
        self.network.reset()
        state4 = self.network.get_state()
       
        
 
        
        

    


    
    
    
    
    
    
if __name__ == '__main__':
    unittest.main()
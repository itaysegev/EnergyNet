
import unittest
import numpy as np

from energy_net.network import Network

from energy_net.entities.network_entity import CompositeNetworkEntity
from energy_net.devices.params import StorageParams, ProductionParams, ConsumptionParams
from energy_net.dynamics.consumption_dynamics.consumption_dynamics import ElectricHeaterDynamics
from energy_net.dynamics.production_dynamics.production_dynamics import PVDynamics
from energy_net.dynamics.storage_dynamics.storage_dynamics import BatteryDynamics

from  energy_net.devices.storage_devices.local_storage import Battery
from  energy_net.devices.consumption_devices.local_consumer import ConsumerDevice
from  energy_net.devices.production_devices.local_producer import PrivateProducer
from energy_net.config import DEFAULT_LIFETIME_CONSTANT

from energy_net.model.action import StorageAction, ProduceAction, ConsumeAction

def default_composite():
    # initialize consumer devices
        consumption_params_arr=[]
        consumption_params = ConsumptionParams(name='pcsunit_consumption', energy_dynamics=ElectricHeaterDynamics(), lifetime_constant=DEFAULT_LIFETIME_CONSTANT)
        consumption_params_arr.append(consumption_params)
        consumption_params_dict = {'pcsunit_consumption': consumption_params}
        
        # initialize storage devices
        storage_params_arr=[]
        storage_params = StorageParams(name = 'test_battery', energy_capacity = 100, power_capacity = 200,initial_charge = 50, charging_efficiency = 1,discharging_efficiency = 1, lifetime_constant = 15, energy_dynamics = BatteryDynamics())
        storage_params_arr.append(storage_params)
        storage_params_dict = {'test_battery': storage_params}

        # initialize production devices
        production_params_arr=[]
        production_params = ProductionParams(name='test_pv', max_production=100, efficiency=0.9, energy_dynamics=PVDynamics())
        production_params_arr.append(production_params)
        production_params_dict = {'test_pv': production_params}

        sub_entities= {name: ConsumerDevice(params) for name, params in consumption_params_dict.items()}
        sub_entities.update({name: Battery(params) for name, params in storage_params_dict.items()})
        sub_entities.update({name: PrivateProducer(params) for name, params in production_params_dict.items()})
        # initilaize pcsunit
        return CompositeNetworkEntity(name="test_composite", sub_entities=sub_entities, agg_func= None)


def default_network():
    strategic_entities = {'pcs_agent': default_composite()}
    return Network(name="test_network", strategic_entities=strategic_entities)


# Write unit tests to test the initialization of the network and the cur_state of the network
class TestNetwork(unittest.TestCase):
    def setUp(self):
        # Assuming `default_network` is a function that returns a default Network instance for testing
        self.network = default_network()

    def test_initialization(self):
        # Test the initialization of your Network
        self.assertEqual(self.network.name, "test_network")
        
        # Test the initialization of the Network's strategic entities
        self.assertEqual(self.network.strategic_entities['pcs_agent'].name, "test_composite")

    def test_observation_space(self):
        # Test the get_observation_space method of your Network
        obs_space = self.network.get_observation_space()
        
        
    def test_action_space(self):
        # Test the get_action_space method of your Network
        action_space = self.network.get_action_space()
        
    
    
    # def test_get_state(self):
    #     # Test the get_state method of your Network
    #     cur_state = self.network.get_state()
    #     self.assertEqual(cur_state['test_composite'].cur_state['pcsunit_consumption'].cur_state['state_of_charge'], 50)
    #     self.assertEqual(cur_state['test_composite'].cur_state['test_battery'].cur_state['state_of_charge'], 50)
    #     self.assertEqual(cur_state['test_composite'].cur_state['test_pv'].cur_state['state_of_charge'], 0)
    #     self.assertEqual(cur_state['network_battery'].cur_state['state_of_charge'], 0)
    
    
    
    
    
    
if __name__ == '__main__':
    unittest.main()
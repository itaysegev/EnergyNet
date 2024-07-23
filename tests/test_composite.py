import unittest
import numpy as np

from energy_net.entities.network_entity import CompositeNetworkEntity
from energy_net.devices.params import StorageParams, ProductionParams, ConsumptionParams
from energy_net.dynamics.consumption_dynamics.consumption_dynamics import GeneralLoad
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
        consumption_params = ConsumptionParams(name='pcsunit_consumption', energy_dynamics=GeneralLoad(), lifetime_constant=DEFAULT_LIFETIME_CONSTANT)
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



class TestComposite(unittest.TestCase):
    def setUp(self):
        # Assuming `default_composite` is a function that returns a default CompositeNetworkEntity instance for testing
        self.composite = default_composite()

    def test_initialization(self):
        # Test the initialization of your CompositeNetworkEntity
        self.assertEqual(self.composite.name, "test_composite")
        
        # # Test the initialization of the CompositeNetworkEntity's devices

        self.assertEqual(self.composite.sub_entities['pcsunit_consumption'].lifetime_constant, DEFAULT_LIFETIME_CONSTANT)
        self.assertEqual(self.composite.sub_entities['test_battery'].energy_capacity, 100)
        self.assertEqual(self.composite.sub_entities['test_pv'].max_production, 100)


    def test_get_state(self):
        # Test the get_current_state method of your CompositeNetworkEntity
        state = self.composite.get_state()
        self.assertEqual(state['pcsunit_consumption']['consumption'], 0.0) 
        self.assertEqual(state['test_battery']['state_of_charge'], 50)
        self.assertEqual(state['test_pv']['production'], 0.0)



    def test_step(self):
        # Test the step method of your CompositeNetworkEntity
        cons_act = ConsumeAction(consume=10)
        storage_act = StorageAction(charge=10)
        prod_act = ProduceAction(produce=10)

        self.composite.step(actions={'pcsunit_consumption': cons_act, 'test_battery': storage_act, 'test_pv': prod_act})
        state = self.composite.get_state()
        self.assertEqual(state['pcsunit_consumption']['consumption'], 10)
        self.assertEqual(state['test_battery']['state_of_charge'], 60)
        self.assertEqual(state['test_pv']['production'], 10)


        cons_act = np.array([10])
        storage_act = np.array([10])
        prod_act = np.array([10])
        self.composite.step(actions={'pcsunit_consumption': cons_act, 'test_battery': storage_act, 'test_pv': prod_act})
        state = self.composite.get_state()
        self.assertEqual(state['pcsunit_consumption']['consumption'], 10)
        self.assertEqual(state['test_battery']['state_of_charge'], 70)
        self.assertEqual(state['test_pv']['production'], 10)


    def test_reset(self):
        # Test the reset method of your CompositeNetworkEntity
        self.composite.reset()
        state = self.composite.get_state()
        self.assertEqual(state['pcsunit_consumption']['consumption'], 0)
        self.assertEqual(state['test_battery']['state_of_charge'], 50)
        self.assertEqual(state['test_pv']['production'], 0)


    def test_get_observation_space(self):
        # Test the get_observation_space method of your CompositeNetworkEntity
        obs_space = self.composite.get_observation_space()
        cons_low = np.array([0, 0, 0])
        cons_high = np.array([np.inf, np.inf, 1.])
        np.testing.assert_array_equal(obs_space['pcsunit_consumption'].low, cons_low)
        np.testing.assert_array_equal(obs_space['pcsunit_consumption'].high, cons_high)

        storage_low = np.array([0., 0., 0., 0., 0., 0.])
        storage_high = np.array([ np.inf,  np.inf, 100.,   1.,   1.,  np.inf])
        np.testing.assert_array_equal(obs_space['test_battery'].low, storage_low)
        np.testing.assert_array_equal(obs_space['test_battery'].high, storage_high)

        prod_low = np.array([0., 0.])
        prod_high = np.array([100.,  np.inf])
        np.testing.assert_array_equal(obs_space['test_pv'].low, prod_low)
        np.testing.assert_array_equal(obs_space['test_pv'].high, prod_high)


    def test_get_action_space(self):
        # Test the get_action_space method of your CompositeNetworkEntity
        action_space = self.composite.get_action_space()
        cons_low = np.array([0])
        cons_high = np.array([np.inf])
        np.testing.assert_array_equal(action_space['pcsunit_consumption'].low, cons_low)
        np.testing.assert_array_equal(action_space['pcsunit_consumption'].high, cons_high)

        storage_low = np.array([-50.])
        storage_high = np.array([50.])
        np.testing.assert_array_equal(action_space['test_battery'].low, storage_low)
        np.testing.assert_array_equal(action_space['test_battery'].high, storage_high)

        prod_low = np.array([0])
        prod_high = np.array([100.])
        np.testing.assert_array_equal(action_space['test_pv'].low, prod_low)
        np.testing.assert_array_equal(action_space['test_pv'].high, prod_high)


if __name__ == '__main__':
    unittest.main()
import unittest
import math

import numpy as np

from energy_net.devices.storage_devices.local_storage import Battery
from energy_net.devices.production_devices.local_producer import PrivateProducer
from energy_net.devices.consumption_devices.local_consumer import ConsumerDevice
from energy_net.dynamics.storage_dynamics import BatteryDynamics
from energy_net.dynamics.production_dynamics import PVDynamics
from energy_net.dynamics.consumption_dynamics import ElectricHeaterDynamics
from energy_net.model.action import EnergyAction
from energy_net.defs import Bounds

from energy_net.config import MIN_CHARGE, MIN_EFFICIENCY, MAX_EFFICIENCY, MIN_CAPACITY, MAX_CAPACITY, INITIAL_TIME, MAX_TIME, MAX_CAPACITY, MIN_CHARGE, MIN_EFFICIENCY, MIN_CAPACITY, MIN_POWER, MAX_ELECTRIC_POWER


class TestBattery(unittest.TestCase):
    def setUp(self):
        storage_params = {
            'energy_capacity': 100,
            'power_capacity': 200,
            'initial_charge': 50,
            'charging_efficiency': 1,
            'discharging_efficiency': 1,
            'lifetime_constant': 15,
            'initial_time': 0,
            'name': 'test_battery',
            'energy_dynamics': BatteryDynamics()
        }
        self.battery = Battery(storage_params)

    def test_initialization(self):
        self.assertEqual(self.battery.energy_capacity, 100)
        self.assertEqual(self.battery.power_capacity, 200)
        self.assertEqual(self.battery.state_of_charge, 50)
        self.assertEqual(self.battery.charging_efficiency, 1)
        self.assertEqual(self.battery.discharging_efficiency, 1)
        self.assertEqual(self.battery.lifetime_constant, 15)


    def test_get_action_space(self):
        action_space = self.battery.get_action_space()
        
        self.assertEqual(type(action_space), Bounds)
        self.assertEqual(action_space.low, -50)
        self.assertEqual(action_space.high, 50)

    def test_get_observation_space(self):
        observation_space = self.battery.get_observation_space()
        self.assertEqual(type(observation_space), Bounds)
        low = np.array([MIN_CAPACITY, MIN_CAPACITY, MIN_CHARGE, MIN_EFFICIENCY, MIN_EFFICIENCY, INITIAL_TIME])
        high = np.array([MAX_CAPACITY, MAX_CAPACITY, 100, MAX_EFFICIENCY, MAX_EFFICIENCY, MAX_TIME])
        np.testing.assert_array_equal(observation_space.low, low)
        np.testing.assert_array_equal(observation_space.high, high)

    def test_step(self):
        state = self.battery.init_state
        self.battery.step(action=EnergyAction(charge=10))
        state['state_of_charge'] = 60
        state['current_time'] = 1
        self.assertEqual(self.battery.state, state)
        
        state['state_of_charge'] = self.battery.energy_capacity
        state['current_time'] = 2
        self.battery.step(action=EnergyAction(charge=150))
        self.assertEqual(self.battery.state, state)
       
  


class TestPrivateProducer(unittest.TestCase):
    def setUp(self):
        producer_params = {
            'max_production': 100,
            'name': 'test_producer',
            'energy_dynamics': PVDynamics()
        }
        self.producer = PrivateProducer(producer_params)

    def test_initialization(self):
        self.assertEqual(self.producer.max_production, 100)

    def test_get_action_space(self):
        action_space = self.producer.get_action_space()
        self.assertEqual(type(action_space), Bounds)
        self.assertEqual(action_space.low, 0)
        self.assertEqual(action_space.high, 100)

    def test_get_observation_space(self):
        observation_space = self.producer.get_observation_space()
        self.assertEqual(type(observation_space), Bounds)
        low = np.array([MIN_POWER, MIN_POWER])
        high = np.array([100, MAX_ELECTRIC_POWER])
        np.testing.assert_array_equal(observation_space.low, low)
        np.testing.assert_array_equal(observation_space.high, high)

    #TODO: Add more tests
    def test_step(self):
        state = self.producer.init_state
        self.producer.step(action=EnergyAction(produce=10))
        state['production'] = 10
        self.assertEqual(self.producer.state, state)
        
        self.producer.step(action=EnergyAction(produce=150))
        self.assertEqual(self.producer.state, state)

    #TODO: Add more tests
    def test_reset(self):
        self.producer.reset()
        self.assertEqual(self.producer.max_production, 100)
        self.assertEqual(self.producer.production, 0)


# class TestConsumerDevice(unittest.TestCase):
#     def setUp(self):
#         consumer_params = {
#             'max_electric_power': 100,
#             'name': 'test_consumer',
#             'energy_dynamics': ElectricHeaterDynamics()
#         }
#         self.consumer = ConsumerDevice(consumer_params)
    
#     def test_initialization(self):
#         self.assertEqual(self.consumer.max_electric_power, 100)

#     def test_get_action_space(self):
#         action_space = self.consumer.get_action_space()
#         self.assertEqual(type(action_space), Bounds)
#         self.assertEqual(action_space.low, 0)
#         self.assertEqual(action_space.high, 100)

#     def test_get_observation_space(self):
#         observation_space = self.consumer.get_observation_space()
#         self.assertEqual(type(observation_space), Bounds)
#         low = np.array([NO_CONSUMPTION, MIN_POWER, MIN_EFFICIENCY])
#         high = np.array([100, MAX_ELECTRIC_POWER, MAX_EFFICIENCY])
#         np.testing.assert_array_equal(observation_space.low, low)
#         np.testing.assert_array_equal(observation_space.high, high)


#     def test_reset(self):
#         self.consumer.reset()
#         self.assertEqual(self.consumer.max_electric_power, 100)
#         self.assertEqual(self.consumer.consumption, NO_CONSUMPTION)

#     def test_step(self):
#         new_consumption = self.consumer.step(action=EnergyAction(consume=10),
#                                             state=dict(consumption=self.consumer.consumption,
#                                                        max_electric_power=self.consumer.max_electric_power
#                                                        ))
#         self.assertEqual(new_consumption, 10)
#         self.consumer.update_consumption(new_consumption)
#         self.assertEqual(self.consumer.consumption, 10)
#         new_consumption = self.consumer.step(action=EnergyAction(consume=-5),
#                                             state=dict(consumption=self.consumer.consumption,
#                                                        max_electric_power=self.consumer.max_electric_power
#                                                        ))
#         self.assertEqual(new_consumption, -5)
#         self.consumer.update_consumption(new_consumption)
#         self.assertEqual(self.consumer.consumption, 5)

    


if __name__ == '__main__':
    unittest.main()
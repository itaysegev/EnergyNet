import random
import unittest
import math

import numpy as np

from energy_net.devices.storage_devices.local_storage import Battery
from energy_net.devices.production_devices.local_producer import PrivateProducer
from energy_net.devices.consumption_devices.local_consumer import ConsumerDevice
from energy_net.dynamics.storage_dynamics.storage_dynamics import BatteryDynamics
from energy_net.dynamics.production_dynamics.production_dynamics import PVDynamics
from energy_net.dynamics.consumption_dynamics.consumption_dynamics import ElectricHeaterDynamics
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


    def test_step_no_losses(self):
        b = self.battery
        initial_state = b.state  # Save initial state for comparison

        # Step with a fixed action
        b.step(action=EnergyAction({'charge': 10}))
        expected_state_after_charge = initial_state + 10  # Adjust this based on actual logic
        self.assertEqual(b.state, expected_state_after_charge)

        # Iterate with random values
        n = 20
        for _ in range(n):
            v = random.uniform(-150, 150)
            previous_state = b.state
            b.step(action=EnergyAction({'charge': v}))
            expected_state_after_random_action = previous_state + v  # Adjust this based on actual logic
            self.assertEqual(b.state, expected_state_after_random_action)

        

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

    def test_step_with_changing_discharging_efficiency(self):
        b = self.battery
        initial_state = b.state  # Save initial state for comparison

        dis_eff = 0.7

        # Change discharging efficiency and validate it
        # b.discharging_efficiency = 0.5 # TODO: Notice that this command doesnt affect the state that is passed to the dynamics.do function
        b.state['discharging_efficiency'] = dis_eff
        self.assertEqual(b.state['discharging_efficiency'], dis_eff)

        # Perform a fixed action with discharging efficiency applied
        b.step(action=EnergyAction({'charge': -10}))
        expected_state_after_charge = initial_state['state_of_charge'] - 10 * dis_eff  # Adjust this based on actual logic
        self.assertEqual(b.state['state_of_charge'], expected_state_after_charge)

        # Iterate with random values and apply efficiency
        n = 20
        for _ in range(n):
            v = random.uniform(-150, 150)
            previous_state_of_charge = b.state['state_of_charge']
            b.step(action=EnergyAction({'charge': v}))
            if v > 0:
                v = v * b.state['charging_efficiency']
            else:
                v = v * b.state['discharging_efficiency']
            expected_state_after_random_action = previous_state_of_charge + v  # Adjust this based on actual logic
            if expected_state_after_random_action > b.state['energy_capacity']:
                expected_state_after_random_action = b.state['energy_capacity']
            elif expected_state_after_random_action < 0:
                expected_state_after_random_action = 0

            self.assertEqual(b.state['state_of_charge'], expected_state_after_random_action)

            if n % 5 == 0:
                b.state['discharging_efficiency'] = dis_eff * 0.5
                self.assertEqual(b.state['discharging_efficiency'], dis_eff * 0.5)


    def test_step_with_changing_charging_efficiency(self):
        b = self.battery
        initial_state = b.state  # Save initial state for comparison

        chg_eff = 0.7

        # Change discharging efficiency and validate it
        # b.discharging_efficiency = 0.5 # TODO: Notice that this command doesnt affect the state that is passed to the dynamics.do function
        b.state['charging_efficiency'] = chg_eff
        self.assertEqual(b.state['charging_efficiency'], chg_eff)

        # Iterate with random values and apply efficiency
        n = 100
        for _ in range(n):
            v = random.uniform(-150, 150)
            previous_state_of_charge = b.state['state_of_charge']
            b.step(action=EnergyAction({'charge': v}))
            if v > 0:
                v = v * b.state['charging_efficiency']
            else:
                v = v * b.state['discharging_efficiency']
            expected_state_after_random_action = previous_state_of_charge + v  # Adjust this based on actual logic
            if expected_state_after_random_action > b.state['energy_capacity']:
                expected_state_after_random_action = b.state['energy_capacity']
            elif expected_state_after_random_action < 0:
                expected_state_after_random_action = 0

            self.assertEqual(b.state['state_of_charge'], expected_state_after_random_action)

            if n % 5 == 0:
                b.state['charging_efficiency'] = chg_eff * 0.5
                self.assertEqual(b.state['charging_efficiency'], chg_eff * 0.5)

    def test_step_with_changing_efficiencies(self):
        b = self.battery
        initial_state = b.state  # Save initial state for comparison

        chg_eff = 0.7
        dis_eff = 0.8

        # Change discharging efficiency and validate it
        # b.discharging_efficiency = 0.5 # TODO: Notice that this command doesnt affect the state that is passed to the dynamics.do function
        b.state['charging_efficiency'] = chg_eff
        self.assertEqual(b.state['charging_efficiency'], chg_eff)

        b.state['discharging_efficiency'] = dis_eff
        self.assertEqual(b.state['discharging_efficiency'], dis_eff)

        # Iterate with random values and apply efficiency
        n = 100
        for _ in range(n):
            v = random.uniform(-150, 150)
            previous_state_of_charge = b.state['state_of_charge']
            b.step(action=EnergyAction({'charge': v}))
            if v > 0:
                v = v * b.state['charging_efficiency']
            else:
                v = v * b.state['discharging_efficiency']
            expected_state_after_random_action = previous_state_of_charge + v  # Adjust this based on actual logic
            if expected_state_after_random_action > b.state['energy_capacity']:
                expected_state_after_random_action = b.state['energy_capacity']
            elif expected_state_after_random_action < 0:
                expected_state_after_random_action = 0

            self.assertEqual(b.state['state_of_charge'], expected_state_after_random_action)

            if n % 5 == 0:
                b.state['charging_efficiency'] = chg_eff * 0.5
                self.assertEqual(b.state['charging_efficiency'], chg_eff * 0.5)

                b.state['discharging_efficiency'] = dis_eff * 0.5
                self.assertEqual(b.state['discharging_efficiency'], dis_eff * 0.5)


    def test_step_with_changing_capacity(self):
        b = self.battery
        initial_state = b.state  # Save initial state for comparison

        new_capacity = 200

        # Change discharging efficiency and validate it
        # b.discharging_efficiency = 0.5 # TODO: Notice that this command doesnt affect the state that is passed to the dynamics.do function
        b.state['energy_capacity'] = new_capacity
        self.assertEqual(b.state['energy_capacity'], new_capacity)


        # Iterate with random values and apply efficiency
        n = 100
        for i in range(n):
            v = random.uniform(-300, 300)
            print('random value ', v)
            previous_state_of_charge = b.state['state_of_charge']
            print('previous_state_of_charge ', previous_state_of_charge)
            b.step(action=EnergyAction({'charge': v}))
            if v > 0:
                v = v * b.state['charging_efficiency']
            else:
                v = v * b.state['discharging_efficiency']
            print('value after losses ', v)
            expected_state_after_random_action = previous_state_of_charge + v  # Adjust this based on actual logic
            if expected_state_after_random_action > b.state['energy_capacity']:
                expected_state_after_random_action = b.state['energy_capacity']
            elif expected_state_after_random_action < 0:
                expected_state_after_random_action = 0

            self.assertEqual(b.state['state_of_charge'], expected_state_after_random_action)


            print(i, b.state['energy_capacity'], b.state['state_of_charge'])
            print('\n\n')

            if n % 5 == 0:
                b.state['energy_capacity'] = new_capacity * 0.9
                self.assertEqual(b.state['energy_capacity'], new_capacity * 0.9)


    def test_step_with_changing_capacity_and_efficiencies(self):
        b = self.battery
        initial_state = b.state  # Save initial state for comparison

        chg_eff = 0.7
        dis_eff = 0.8
        new_capacity = 80

        # Change discharging efficiency and validate it
        # b.discharging_efficiency = 0.5 # TODO: Notice that this command doesnt affect the state that is passed to the dynamics.do function
        b.state['charging_efficiency'] = chg_eff
        self.assertEqual(b.state['charging_efficiency'], chg_eff)

        b.state['discharging_efficiency'] = dis_eff
        self.assertEqual(b.state['discharging_efficiency'], dis_eff)

        # Change discharging efficiency and validate it
        # b.discharging_efficiency = 0.5 # TODO: Notice that this command doesnt affect the state that is passed to the dynamics.do function
        b.state['energy_capacity'] = new_capacity
        self.assertEqual(b.state['energy_capacity'], new_capacity)

        # Iterate with random values and apply efficiency
        n = 100
        for _ in range(n):
            v = random.uniform(-150, 150)
            previous_state_of_charge = b.state['state_of_charge']
            b.step(action=EnergyAction({'charge': v}))
            if v > 0:
                v = v * b.state['charging_efficiency']
            else:
                v = v * b.state['discharging_efficiency']
            expected_state_after_random_action = previous_state_of_charge + v  # Adjust this based on actual logic
            if expected_state_after_random_action > b.state['energy_capacity']:
                expected_state_after_random_action = b.state['energy_capacity']
            elif expected_state_after_random_action < 0:
                expected_state_after_random_action = 0

            self.assertEqual(b.state['state_of_charge'], expected_state_after_random_action)

            if n % 5 == 0:
                b.state['charging_efficiency'] = chg_eff * 0.5
                self.assertEqual(b.state['charging_efficiency'], chg_eff * 0.5)

                b.state['discharging_efficiency'] = dis_eff * 0.5
                self.assertEqual(b.state['discharging_efficiency'], dis_eff * 0.5)

                b.state['energy_capacity'] = new_capacity * 0.9
                self.assertEqual(b.state['energy_capacity'], new_capacity * 0.9)


    def test_step_with_changing_lifetime(self):
        b = self.battery
        initial_state = b.state  # Save initial state for comparison

        decay_constant = 0.1

        b.lifetime_constant = decay_constant

        # Iterate with random values and apply efficiency
        n = 100
        for _ in range(n):
            v = random.uniform(-150, 150)
            previous_state_of_charge = b.state['state_of_charge']
            b.step(action=EnergyAction({'charge': v}), params=b.lifetime_constant)
            if v > 0:
                v = v * b.state['charging_efficiency']
            else:
                v = v * b.state['discharging_efficiency']
            expected_state_after_random_action = previous_state_of_charge + v  # Adjust this based on actual logic
            if expected_state_after_random_action > b.state['energy_capacity']:
                expected_state_after_random_action = b.state['energy_capacity']
            elif expected_state_after_random_action < 0:
                expected_state_after_random_action = 0

            expected_state_after_random_action = expected_state_after_random_action * np.exp(-decay_constant) # decay

            self.assertEqual(b.state['state_of_charge'], expected_state_after_random_action)


       
# class TestPrivateProducer(unittest.TestCase):
#     def setUp(self):
#         producer_params = {
#             'max_production': 100,
#             'name': 'test_producer',
#             'energy_dynamics': PVDynamics()
#         }
#         self.producer = PrivateProducer(producer_params)

#     def test_initialization(self):
#         self.assertEqual(self.producer.max_production, 100)

#     def test_get_action_space(self):
#         action_space = self.producer.get_action_space()
#         self.assertEqual(type(action_space), Bounds)
#         self.assertEqual(action_space.low, 0)
#         self.assertEqual(action_space.high, 100)

#     def test_get_observation_space(self):
#         observation_space = self.producer.get_observation_space()
#         self.assertEqual(type(observation_space), Bounds)
#         low = np.array([MIN_POWER, MIN_POWER])
#         high = np.array([100, MAX_ELECTRIC_POWER])
#         np.testing.assert_array_equal(observation_space.low, low)
#         np.testing.assert_array_equal(observation_space.high, high)

#     #TODO: Add more tests
#     def test_step(self):
#         state = self.producer.init_state
#         self.producer.step(action=EnergyAction(produce=10))
#         state['production'] = 10
#         self.assertEqual(self.producer.state, state)
        
#         self.producer.step(action=EnergyAction(produce=150))
#         self.assertEqual(self.producer.state, state)

#     #TODO: Add more tests
#     def test_reset(self):
#         self.producer.reset()
#         self.assertEqual(self.producer.max_production, 100)
#         self.assertEqual(self.producer.production, 0)


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
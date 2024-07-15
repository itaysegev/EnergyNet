import copy
import random
import unittest
import math

import numpy as np

from energy_net.devices.storage_devices.local_storage import Battery

from energy_net.dynamics.storage_dynamics.storage_dynamics import BatteryDynamics

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


    def dynamics_loop_test(self, b, bound, chg_eff,  dis_eff, capacity, decay_constant=None):
        v = random.uniform(-bound, bound)
        previous_state_of_charge = b.cur_state['state_of_charge']
        if decay_constant is None:
            b.step(action=EnergyAction({'charge': v}))
        else:
            b.step(action=EnergyAction({'charge': v}), lifetime_constant=decay_constant)
        if v > 0:
            v = v * chg_eff
        else:
            v = v * dis_eff
        expected_state_after_random_action = previous_state_of_charge + v  # Adjust this based on actual logic
        if expected_state_after_random_action > capacity:
            expected_state_after_random_action = capacity
        elif expected_state_after_random_action < 0:
            expected_state_after_random_action = 0
        return expected_state_after_random_action


    def test_step_no_losses(self):
        b = copy.deepcopy(self.battery)

        # Device properties
        chg_eff = b.cur_state['discharging_efficiency']
        dis_eff = b.cur_state['discharging_efficiency']
        capacity = b.cur_state['energy_capacity']

        bound = 1.2 * b.energy_capacity

        # Iterate with random values and apply efficiency
        n = 100
        for _ in range(n):
            expected_state_after_random_action = self.dynamics_loop_test(b=b, bound=bound, chg_eff=chg_eff, dis_eff=dis_eff, capacity=capacity)
            self.assertEqual(b.cur_state['state_of_charge'], expected_state_after_random_action)


    def test_step_with_changing_discharging_efficiency(self):
        b = copy.deepcopy(self.battery)

        # Device properties
        chg_eff = b.cur_state['discharging_efficiency']
        dis_eff = b.cur_state['discharging_efficiency']
        capacity = b.cur_state['energy_capacity']

        bound = 1.2 * b.energy_capacity

        # Iterate with random values and apply efficiency
        n = 100
        for i in range(n):
            expected_state_after_random_action = self.dynamics_loop_test(b=b, bound=bound, chg_eff=chg_eff,
                                                                         dis_eff=dis_eff, capacity=capacity)
            self.assertEqual(b.cur_state['state_of_charge'], expected_state_after_random_action)

            # Update discharge efficiency
            if i % 5 == 0:
                dis_eff = dis_eff * 0.5
                b.cur_state['discharging_efficiency'] = dis_eff
                self.assertEqual(b.cur_state['discharging_efficiency'], dis_eff)

    def test_step_with_changing_charging_efficiency(self):
        b = copy.deepcopy(self.battery)

        # Device properties
        chg_eff = b.cur_state['discharging_efficiency']
        dis_eff = b.cur_state['discharging_efficiency']
        capacity = b.cur_state['energy_capacity']

        bound = 1.2 * b.energy_capacity

        # Iterate with random values and apply efficiency
        n = 100
        for i in range(n):
            expected_state_after_random_action = self.dynamics_loop_test(b=b, bound=bound, chg_eff=chg_eff,
                                                                         dis_eff=dis_eff, capacity=capacity)
            self.assertEqual(b.cur_state['state_of_charge'], expected_state_after_random_action)

            # Update charging efficiency
            if i % 5 == 0:
                chg_eff = chg_eff * 0.5
                b.cur_state['charging_efficiency'] = chg_eff
                self.assertEqual(b.cur_state['charging_efficiency'], chg_eff)

    def test_step_with_changing_efficiencies(self):
        b = copy.deepcopy(self.battery)

        # Device properties
        chg_eff = b.cur_state['discharging_efficiency']
        dis_eff = b.cur_state['discharging_efficiency']
        capacity = b.cur_state['energy_capacity']

        bound = 1.2 * capacity

        # Iterate with random values and apply efficiency
        n = 100
        for i in range(n):
            expected_state_after_random_action = self.dynamics_loop_test(b=b, bound=bound, chg_eff=chg_eff,
                                                                         dis_eff=dis_eff, capacity=capacity)
            self.assertEqual(b.cur_state['state_of_charge'], expected_state_after_random_action)

            # Update charging efficiency
            if i % 5 == 0:
                chg_eff = chg_eff * 0.5
                b.cur_state['charging_efficiency'] = chg_eff
                self.assertEqual(b.cur_state['charging_efficiency'], chg_eff)

                dis_eff = dis_eff * 0.5
                b.cur_state['discharging_efficiency'] = dis_eff
                self.assertEqual(b.cur_state['discharging_efficiency'], dis_eff)


    def test_step_with_changing_capacity(self):
        b = copy.deepcopy(self.battery)

        # Device properties
        chg_eff = b.cur_state['discharging_efficiency']
        dis_eff = b.cur_state['discharging_efficiency']
        capacity = b.cur_state['energy_capacity']

        bound = 1.2 * capacity

        # Iterate with random values and apply efficiency
        n = 100
        for i in range(n):
            expected_state_after_random_action = self.dynamics_loop_test(b=b, bound=bound, chg_eff=chg_eff,
                                                                         dis_eff=dis_eff, capacity=capacity)
            self.assertEqual(b.cur_state['state_of_charge'], expected_state_after_random_action)

            # Update charging efficiency
            if i % 5 == 0:
                capacity = capacity * 0.9
                b.cur_state['energy_capacity'] = capacity
                self.assertEqual(b.cur_state['energy_capacity'], capacity)


    def test_step_with_changing_capacity_and_efficiencies(self):
        b = copy.deepcopy(self.battery)

        # Device properties
        chg_eff = b.cur_state['discharging_efficiency']
        dis_eff = b.cur_state['discharging_efficiency']
        capacity = b.cur_state['energy_capacity']

        bound = 1.2 * capacity

        # Iterate with random values and apply efficiency
        n = 100
        for i in range(n):
            expected_state_after_random_action = self.dynamics_loop_test(b=b, bound=bound, chg_eff=chg_eff,
                                                                         dis_eff=dis_eff, capacity=capacity)
            self.assertEqual(b.cur_state['state_of_charge'], expected_state_after_random_action)

            # Update efficiencies and capacity
            if i == 20:
                chg_eff = chg_eff * 0.5
                dis_eff = dis_eff * 0.5
                capacity = capacity * 0.97

                b.cur_state['charging_efficiency'] = chg_eff
                self.assertEqual(b.cur_state['charging_efficiency'], chg_eff)

                b.cur_state['discharging_efficiency'] = dis_eff
                self.assertEqual(b.cur_state['discharging_efficiency'], dis_eff)

                b.cur_state['energy_capacity'] = capacity
                self.assertEqual(b.cur_state['energy_capacity'], capacity)


    def test_step_with_changing_lifetime(self):
        b = copy.deepcopy(self.battery)

        # Device properties
        chg_eff = b.cur_state['discharging_efficiency']
        dis_eff = b.cur_state['discharging_efficiency']
        capacity = b.cur_state['energy_capacity']

        bound = 1.2 * capacity

        decay_constant = 0.01

        b.lifetime_constant = decay_constant

        # Iterate with random values and apply efficiency
        n = 20
        for i in range(n):
            expected_state_after_random_action = self.dynamics_loop_test(b=b, bound=bound, chg_eff=chg_eff,
                                                                         dis_eff=dis_eff, capacity=capacity,
                                                                         decay_constant=decay_constant)
            self.assertEqual(b.cur_state['state_of_charge'], expected_state_after_random_action)

            exponent = i / float(decay_constant)
            exponent = max(-200, min(200, exponent))
            capacity = capacity * np.exp(-exponent)

            self.assertEqual(b.cur_state['energy_capacity'], capacity)

       
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
#         cur_state = self.producer.init_state
#         self.producer.step(action=EnergyAction(produce=10))
#         cur_state['production'] = 10
#         self.assertEqual(self.producer.cur_state, cur_state)
        
#         self.producer.step(action=EnergyAction(produce=150))
#         self.assertEqual(self.producer.cur_state, cur_state)

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
#                                             cur_state=dict(consumption=self.consumer.consumption,
#                                                        max_electric_power=self.consumer.max_electric_power
#                                                        ))
#         self.assertEqual(new_consumption, 10)
#         self.consumer.update_consumption(new_consumption)
#         self.assertEqual(self.consumer.consumption, 10)
#         new_consumption = self.consumer.step(action=EnergyAction(consume=-5),
#                                             cur_state=dict(consumption=self.consumer.consumption,
#                                                        max_electric_power=self.consumer.max_electric_power
#                                                        ))
#         self.assertEqual(new_consumption, -5)
#         self.consumer.update_consumption(new_consumption)
#         self.assertEqual(self.consumer.consumption, 5)

    


if __name__ == '__main__':
    unittest.main()

import sys
import os
import warnings
# Add the project's root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../energy_net')))

import unittest
from defs import EnergyAction, ChargeAction, ProduceAction, ConsumeAction
from entities.local_storage import Battery
from dynamics.storage_dynamics import BatteryDynamics
from entities.private_producer import PrivateProducer
from dynamics.production_dynmaics import PVDynamics
from entities.local_consumer import ConsumerDevice
from dynamics.consumption_dynamic import ElectricHeaterDynamics
from entities.pcsunit import pcsunit

from env.EnergyNetEnv import EnergyNetEnv
from pettingzoo.test import  parallel_api_test, performance_benchmark, parallel_seed_test
from pettingzoo.test.api_test import missing_attr_warning

class TestEnergyNetEnv(unittest.TestCase):
    def setUp(self):
        self.battery = Battery(capacity=100, efficiency=0.9, energy_dynamics=BatteryDynamics(), name='test_battery')
        self.pv = PrivateProducer(max_produce=100, efficiency=0.9, energy_dynamics=PVDynamics(), name='test_pv')
        self.load = ConsumerDevice(efficiency=0.9, max_electric_power=100, energy_dynamics=ElectricHeaterDynamics(), name='test_heater')
        self.pcsunit = pcsunit(name='test_pcsunit', sub_entities=[self.battery, self.pv, self.load], agg_func=lambda x: x)
        self.env = EnergyNetEnv(network_entities=[self.pcsunit], 
            simulation_start_time_step=0,
            simulation_end_time_step=100, 
            episode_time_steps=100, 
            seconds_per_time_step=1, 
            reward_function=None, 
            random_seed=None)

    def test_reset(self):
        self.env.reset()

    def test_step(self):
        self.env.reset()
        
        action = {'test_pcsunit': EnergyAction()}
        self.env.step(action)

    def test_observe_all(self):
        self.env.reset()
        self.env.observe_all()

    def test_get_observation_space(self):
        self.env.observation_space(self.env.entities['test_pcsunit'].name)

    def test_get_action_space(self):
        self.env.action_space(self.env.entities['test_pcsunit'].name)

    
    def test_parallel_api(self):
        self.env.reset()
        parallel_api_test(self.env, num_cycles=1000)
        

    # def test_performance_benchmark(self):
    #     self.env.reset()
    #     performance_benchmark(self.env)


    # def test_parallel_seed_test(self):
    #     def parallel_env_fn():
    #         return self.env
    #     self.env.reset()
    #     parallel_seed_test(parallel_env_fn=parallel_env_fn)


class TestBattery(unittest.TestCase):
    def setUp(self):
        self.battery = Battery(energy_capacity = 100, power_capacity = 200,
                    state_of_charge = 50, charging_efficiency = 1,
                    discharging_efficiency = 1, lifetime_constant = 15, energy_dynamics=BatteryDynamics(), name='test_battery')

    def test_initialization(self):
        self.assertEqual(self.battery.energy_capacity, 100)
        self.assertEqual(self.battery.power_capacity, 200)
        self.assertEqual(self.battery.state_of_charge, 50)
        self.assertEqual(self.battery.charging_efficiency, 1)
        self.assertEqual(self.battery.discharging_efficiency, 1)
        self.assertEqual(self.battery.lifetime_constant, 15)

    def test_energy_dynamics(self):
        new_state_of_charge = self.battery.step(action=EnergyAction(charge=10),
                                                state=dict(state_of_charge=self.battery.state_of_charge,
                                                           energy_capacity=self.battery.energy_capacity,
                                                           power_capacity=self.battery.power_capacity,
                                                           charging_efficiency=self.battery.charging_efficiency,
                                                           discharging_efficiency=self.battery.discharging_efficiency,
                                                           lifetime_constant=self.battery.lifetime_constant
                                                           ))
        self.assertEqual(new_state_of_charge, 10)
        self.battery.update_state_of_charge(new_state_of_charge)
        new_energy_capacity = 100 * math.exp(-1/self.battery.lifetime_constant)
        new_power_capacity = 200 * math.exp(-1 / self.battery.lifetime_constant)
        new_charging_efficiency = 1 * math.exp(-1/self.battery.lifetime_constant)
        new_discharging_efficiency = 1 * math.exp(-1/self.battery.lifetime_constant)
        self.assertEqual(self.battery.energy_capacity, new_energy_capacity)
        self.assertEqual(self.battery.power_capacity, new_power_capacity)
        self.assertEqual(self.battery.state_of_charge, 60)
        self.assertEqual(self.battery.charging_efficiency, new_charging_efficiency)
        self.assertEqual(self.battery.discharging_efficiency, new_discharging_efficiency)
        self.assertEqual(self.battery.lifetime_constant, 15)
        new_state_of_charge = self.battery.step(action=EnergyAction(charge=-5),
                                                state=dict(state_of_charge=self.battery.state_of_charge,
                                                           energy_capacity=self.battery.energy_capacity,
                                                           power_capacity=self.battery.power_capacity,
                                                           charging_efficiency=self.battery.charging_efficiency,
                                                           discharging_efficiency=self.battery.discharging_efficiency,
                                                           lifetime_constant=self.battery.lifetime_constant
                                                           ))
        self.assertEqual(new_state_of_charge, -5)
        self.battery.update_state_of_charge(new_state_of_charge)
        new_energy_capacity = new_energy_capacity * math.exp(-1 / self.battery.lifetime_constant)
        new_power_capacity = new_power_capacity * math.exp(-1 / self.battery.lifetime_constant)
        new_charging_efficiency = new_charging_efficiency * math.exp(-1 / self.battery.lifetime_constant)
        new_discharging_efficiency = new_discharging_efficiency * math.exp(-1 / self.battery.lifetime_constant)
        self.assertEqual(self.battery.energy_capacity, new_energy_capacity)
        self.assertEqual(self.battery.power_capacity, new_power_capacity)
        self.assertEqual(self.battery.state_of_charge, 55)
        self.assertEqual(self.battery.charging_efficiency, new_charging_efficiency)
        self.assertEqual(self.battery.discharging_efficiency, new_discharging_efficiency)
        self.assertEqual(self.battery.lifetime_constant, 15)  




if __name__ == '__main__':
    unittest.main()
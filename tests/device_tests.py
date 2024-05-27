import sys
import os

# Add the project's root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../energy_net')))

import unittest
from defs import EnergyAction
from entities.local_storage import Battery
from dynamics.storage_dynamics import BatteryDynamics
from entities.private_producer import PrivateProducer
from dynamics.production_dynmaics import PVDynamics
from entities.local_consumer import ConsumerDevice
from dynamics.consumption_dynamic import ElectricHeaterDynamics
from entities.pcsunit import pcsunit
import numpy as np

from env.EnergyNetEnv import EnergyNetEnv

class TestBattery(unittest.TestCase):
    def setUp(self):
        self.battery = Battery(capacity=100, efficiency=0.9, energy_dynamics=BatteryDynamics(), name='test_battery')

    def test_initialization(self):
        self.assertEqual(self.battery.capacity, 100)

    def test_energy_dynamics(self):
        self.battery.step(action = np.array([10]))
        self.assertEqual(self.battery.capacity, 100)
        self.assertEqual(self.battery.state_of_charge, 10)
        with self.assertRaises(AssertionError) as context:
            self.battery.step(action = np.array([100]))
        self.assertEqual(str(context.exception), "state_of_charge must be <= capacity.")
        self.battery.step(action = np.array([90]))
        self.assertEqual(self.battery.state_of_charge, 100)
        self.assertEqual(self.battery.capacity, 100)


class TestProducer(unittest.TestCase):
    def setUp(self):
        self.pv = PrivateProducer(max_produce=100, efficiency=0.9, energy_dynamics=PVDynamics(), name='test_pv')
    
    def test_energy_dynamics(self):
        self.pv.step(np.array([90]))
        self.assertEqual(self.pv.max_produce, 81)
        self.pv.step(np.array([100]))
        self.assertEqual(self.pv.max_produce, 90)

class TestConsumer(unittest.TestCase):
    def setUp(self):
        self.heater = ConsumerDevice(efficiency=0.9, max_electric_power=100, energy_dynamics=ElectricHeaterDynamics(), name='test_heater')

    def test_initialization(self):
        self.assertEqual(self.heater.max_electric_power, 100)

    def test_efficiency(self):
        self.assertEqual(self.heater.efficiency, 0.9)
        
    def test_energy_dynamics(self):
        self.heater.step(np.array([80]))
        self.assertEqual(self.heater.max_electric_power, 80 * 0.9)
        self.heater.step(np.array([100]))
        self.assertEqual(self.heater.max_electric_power, 90)

class Testpcsunit(unittest.TestCase):
    def setUp(self):
        self.battery = Battery(capacity=100, efficiency=0.9, energy_dynamics=BatteryDynamics(), name='test_battery')
        self.pv = PrivateProducer(max_produce=100, efficiency=0.9, energy_dynamics=PVDynamics(), name='test_pv')
        self.load = ConsumerDevice(efficiency=0.9, max_electric_power=100, energy_dynamics=ElectricHeaterDynamics(), name='test_heater')
        self.pcsunit = pcsunit(name='test_pcsunit', sub_entities=[self.battery, self.pv, self.load], agg_func=lambda x: x)

    def test_step(self):
        self.pcsunit.step({'test_battery': np.array([90]), 'test_heater': np.array([10]), 'test_pv': np.array([100])})
        self.assertEqual(self.pcsunit.sub_entities['test_battery'].state_of_charge, 90)



        
    

if __name__ == '__main__':
    unittest.main()


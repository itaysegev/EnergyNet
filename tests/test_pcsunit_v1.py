import unittest
from energy_net.entities.pcsunit import PCSUnit
from energy_net.devices.params import StorageParams, ProductionParams, ConsumptionParams
from energy_net.dynamics.consumption_dynamics.consumption_dynamics import ConsumptionDynamics
from energy_net.dynamics.production_dynamics.production_dynamics import PVDynamics
from energy_net.dynamics.storage_dynamics.storage_dynamics import BatteryDynamics
from energy_net.config import DEFAULT_LIFETIME_CONSTANT
import numpy as np

def default_pcsunit():
    # initialize consumer devices
        consumption_params_arr=[]
        consumption_params = ConsumptionParams(name='pcsunit_consumption', energy_dynamics=ConsumptionDynamics(), lifetime_constant=DEFAULT_LIFETIME_CONSTANT)
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

        # initilaize pcsunit
        return PCSUnit(name="test_pcsunit", consumption_params_dict=consumption_params_dict, storage_params_dict=storage_params_dict, production_params_dict=production_params_dict, agg_func= None)


class TestPCSUnit(unittest.TestCase):
    def setUp(self):
        # Assuming `default_pcsunit` is a function that returns a default PCSUnit instance for testing
        self.pcs_unit = default_pcsunit()

    def test_initialization(self):
        # Test the initialization of your PCSUnit
        self.assertEqual(self.pcs_unit.name, "test_pcsunit")
        

        # Test the initialization of the PCSUnit's devices
        self.assertEqual(self.pcs_unit.sub_entities[self.pcs_unit.consumption_keys[0]].lifetime_constant, DEFAULT_LIFETIME_CONSTANT)
        self.assertEqual(self.pcs_unit.sub_entities[self.pcs_unit.storage_keys[0]].energy_capacity, 100)
        self.assertEqual(self.pcs_unit.sub_entities[self.pcs_unit.production_keys[0]].max_production, 100)

        


if __name__ == '__main__':
    unittest.main()
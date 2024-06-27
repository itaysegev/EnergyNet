
import json
from pathlib import Path
from typing import List

from energy_net.entities.params import StorageParams, ProductionParams, ConsumptionParams
from energy_net.entities.pcsunit import PCSUnit
from energy_net.dynamics.consumption_dynamics import PCSUnitConsumptionDynamics
from energy_net.network_entity import NetworkEntity
from energy_net.config import DEFAULT_LIFETIME_CONSTANT
from energy_net.dynamics.storage_dynamics import BatteryDynamics
from energy_net.dynamics.production_dynamics import PVDynamics

def example_pcsunit():
    # initialize consumer devices
        consumption_params_arr=[]
        consumption_params = ConsumptionParams(name='pcsunit_consumption', energy_dynamics=PCSUnitConsumptionDynamics(), lifetime_constant=DEFAULT_LIFETIME_CONSTANT)
        consumption_params_arr.append(consumption_params)
        consumption_params_dict = {'pcsunit_consumption': consumption_params}

        # initialize storage devices
        storage_params_arr=[]
        storage_params = StorageParams(name = 'test_battery', energy_capacity = 100, power_capacity = 200,inital_charge = 50, charging_efficiency = 1,discharging_efficiency = 1, lifetime_constant = 15, energy_dynamics = BatteryDynamics())
        storage_params_arr.append(storage_params)
        storage_params_dict = {'test_battery': storage_params}

        # initialize production devices
        production_params_arr=[]
        production_params = ProductionParams(name='test_pv', max_production=100, efficiency=0.9, energy_dynamics=PVDynamics())
        production_params_arr.append(production_params)
        production_params_dict = {'test_pv': production_params}

        # initilaize pcsunit
        return PCSUnit(name="test_pcsunit", consumption_params_dict=consumption_params_dict, storage_params_dict=storage_params_dict, production_params_dict=production_params_dict, agg_func= None)


def default_network_entities() -> List[NetworkEntity]:
        pcsunit = example_pcsunit()
        return [pcsunit]

ENV_CFG_FILE = Path(__file__).parent / 'test_env_configs.json'


def get_env_cfgs():
    with open(ENV_CFG_FILE, 'r') as f:
        env_cfgs = json.load(f)
    env_cfgs['single_entity_simple']['network_entities'] = default_network_entities()
    return env_cfgs


test_env_cfgs = get_env_cfgs()

single_agent_cfgs = {k: cfg for k, cfg in test_env_cfgs.items() if 'network_entities' not in cfg or len(cfg['network_entities']) == 1}

import numpy as np
from gymnasium.spaces import Box
from typing import List, Mapping, Any, Union
import time

from ..defs import Bounds
from ..config import DEFAULT_EFFICIENCY, DEFAULT_LIFETIME_CONSTANT
from ..entities.params import StorageParams, ProductionParams, ConsumptionParams
from ..network_entity import NetworkEntity
from ..entities.local_storage import Battery
from ..dynamics.storage_dynamics import BatteryDynamics
from ..entities.local_producer import PrivateProducer
from ..dynamics.production_dynamics import PVDynamics
from ..entities.consumer_device import ConsumerDevice
from ..dynamics.consumption_dynamics import PCSUnitConsumptionDynamics
from ..entities.pcsunit import PCSUnit
from ..model.reward import RewardFunction




def observation_seperator(observation:dict[str, np.ndarray]):
    """
    Seperates the observation into the agents's observation.

    Parameters:
    observation (dict): The observation of all agents.
    agents (str): The agents to get the observation for.

    Returns:
    dict: The observation of the agents.
    """

    return [observation[name] for name in observation.keys()]


def bounds_to_gym_box(bounds: Bounds) -> Box:
  return Box(
        low=bounds['low'],
        high=bounds['high'],
        shape=bounds['shape'],
        dtype=bounds['dtype']
    )


def default_pcsunit():
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
        pcsunit = default_pcsunit()
        return [pcsunit]





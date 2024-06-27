
import numpy as np
from gymnasium.spaces import Box
from typing import List, Mapping, Any, Union
import time

from ..defs import Bounds
from ..config import DEFAULT_EFFICIENCY, DEFAULT_LIFETIME_CONSTANT
from ..entities.params import StorageParams, ProductionParams, ConsumptionParams
from ..entities.network_entity import NetworkEntity
from ..devices.storage_devices.local_storage import Battery
from ..dynamics.storage_dynamics import BatteryDynamics
from ..devices.production_devices.local_producer import PrivateProducer
from ..dynamics.production_dynamics import PVDynamics
from ..devices.consumer_devices.consumer_device import ConsumerDevice
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


def default_network_entities() -> List[NetworkEntity]:
        pcsunit = default_pcsunit()
        return [pcsunit]





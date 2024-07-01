from typing import TypedDict

from ..config import DEFAULT_LIFETIME_CONSTANT, DEFAULT_EFFICIENCY, DEFAULT_INIT_POWER, DEFAULT_SELF_CONSUMPTION
from ..dynamics.energy_dynamcis import EnergyDynamics


class DeviceParams(TypedDict):
    name: str
    lifetime_constant: float = DEFAULT_LIFETIME_CONSTANT
    max_electric_power: float = DEFAULT_EFFICIENCY
    init_max_electric_power: float = DEFAULT_INIT_POWER
    consumption: float  = DEFAULT_SELF_CONSUMPTION
    efficiency: float = DEFAULT_EFFICIENCY
    energy_dynamics: EnergyDynamics = None


class StorageParams(DeviceParams):
    energy_capacity: float
    power_capacity: float
    initial_charge: float
    charging_efficiency: float
    discharging_efficiency: float
    net_connection_size: float

class ConsumptionParams(DeviceParams):
    energy_capacity: float
    power_capacity: float
    initial_charge: float


class ProductionParams(DeviceParams):
    max_producion: float



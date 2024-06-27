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




'''
    energy_capacity : float, default: inf
        Maximum amount of energy the storage device can store in [kWh]. Must be >= 0.
    power_capacity : float, default: inf
        Maximum amount of power the storage device can store in [kW]. Must be >= 0.
    state_of_charge : float, default: 0.0
        Current state of charge of the storage device in [kWh]. Must be >= 0.
    charging_efficiency : float, default: 1.0
        Charging efficiency of the storage device. Must be > 0.
    discharging_efficiency : float, default: 1.0
        Discharging efficiency of the storage device. Must be > 0.
    lifetime_constant : float, default: inf
        Lifetime constant of the storage device in years. Must be > 0.
    energy_dynamics : Dynamics, default: None
        Energy dynamics of the storage device. Must be a subclass of Dynamics.
    name : str, default: None
        Name of the storage device. Must be a string.
'''

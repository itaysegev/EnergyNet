"""
Parameter definitions for smart grid devices.

This module defines the parameter structures for various devices within the smart grid system
using TypedDicts for type hinting and validation.
"""

from typing import Optional, TypedDict

from ..config import (
    DEFAULT_LIFETIME_CONSTANT,
    DEFAULT_EFFICIENCY,
    DEFAULT_INIT_POWER,
    DEFAULT_SELF_CONSUMPTION,
)

from ..dynamics.energy_dynamcis import EnergyDynamics



class DeviceParams(TypedDict, total=False):
    """
    Base parameters for all devices in the smart grid system.

    Attributes
    ----------
    name : str
        The name of the device.
    lifetime_constant : float
        The lifetime constant representing the device's technical efficiency.
        Defaults to `DEFAULT_LIFETIME_CONSTANT`.
    max_electric_power : float
        The maximum electric power the device can handle in [kW].
        Defaults to `DEFAULT_EFFICIENCY`.
    init_max_electric_power : float
        The initial maximum electric power in [kW].
        Defaults to `DEFAULT_INIT_POWER`.
    consumption : float
        The initial consumption value in [kW].
        Defaults to `DEFAULT_SELF_CONSUMPTION`.
    efficiency : float
        The efficiency of the device.
        Defaults to `DEFAULT_EFFICIENCY`.
    energy_dynamics : Optional[EnergyDynamics]
        The energy dynamics configuration for the device.
        Defaults to `None`.
    """

    name: str
    lifetime_constant: Optional[float] = DEFAULT_LIFETIME_CONSTANT
    max_electric_power: Optional[float] = DEFAULT_EFFICIENCY
    init_max_electric_power: Optional[float] = DEFAULT_INIT_POWER
    consumption: Optional[float] = DEFAULT_SELF_CONSUMPTION
    efficiency: Optional[float] = DEFAULT_EFFICIENCY
    energy_dynamics: Optional[EnergyDynamics] = None


class StorageParams(DeviceParams, total=False):
    """
    Parameters specific to storage devices.

    Attributes
    ----------
    energy_capacity : float
        The maximum energy capacity of the storage device in [kWh].
    power_capacity : float
        The maximum power capacity of the storage device in [kW].
    initial_charge : float
        The initial state of charge of the storage device in [kWh].
    charging_efficiency : float
        The efficiency during the charging process.
    discharging_efficiency : float
        The efficiency during the discharging process.
    net_connection_size : float
        The net connection size of the storage device in [kW].
    """

    energy_capacity: float
    power_capacity: float
    initial_charge: float
    charging_efficiency: float
    discharging_efficiency: float
    net_connection_size: float


class ConsumptionParams(DeviceParams, total=False):
    """
    Parameters specific to consumption devices.

    Attributes
    ----------
    energy_capacity : float
        The energy capacity related to consumption in [kWh].
    power_capacity : float
        The power capacity related to consumption in [kW].
    initial_charge : float
        The initial charge related to consumption in [kWh].
    """

    energy_capacity: float
    power_capacity: float
    initial_charge: float


class ProductionParams(DeviceParams, total=False):
    """
    Parameters specific to production devices.

    Attributes
    ----------
    max_production : float
        The maximum production capacity of the device in [kW].
    """

    max_production: float

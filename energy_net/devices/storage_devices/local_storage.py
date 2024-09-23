"""
This code is based on https://github.com/intelligent-environments-lab/CityLearn/blob/master/citylearn/energy_model.py
"""

import numpy as np
from typing import Any

from ..device import Device
from ..params import DeviceParams, StorageParams
from ...defs import Bounds
from ...model.action import StorageAction
from ...model.state import State, StorageState
from ...config import (
    DEFAULT_EFFICIENCY,
    MIN_CHARGE,
    MIN_EFFICIENCY,
    MAX_EFFICIENCY,
    MIN_CAPACITY,
    MAX_CAPACITY,
    INITIAL_TIME,
    MAX_TIME,
    NO_CHARGE,
)


class Battery(Device):
    """
    Represents an electricity storage device (Battery) in the smart grid system.

    This class models a battery that can store and release electric power, managing its state of charge
    along with charging and discharging efficiencies.

    Parameters
    ----------
    storage_params : StorageParams
        Parameters defining the storage behavior of the battery, including initial charge,
        charging/discharging efficiencies, and capacity limits.

    Attributes
    ----------
    state_of_charge : float
        Current state of charge of the battery in [kWh].
    charging_efficiency : float
        Technical efficiency of the charging process.
    discharging_efficiency : float
        Technical efficiency of the discharging process.
    energy_capacity : float
        Maximum amount of energy the battery can store in [kWh].
    power_capacity : float
        Maximum amount of power the battery can handle in [kW].
    init_time : int
        Initial time step for the battery.
    action_type : StorageAction
        The type of action the battery can perform.
    """

    def __init__(self, storage_params: StorageParams):
        self._state_of_charge = storage_params.get("initial_charge", NO_CHARGE)
        self._charging_efficiency = storage_params.get("charging_efficiency", DEFAULT_EFFICIENCY)
        self._discharging_efficiency = storage_params.get("discharging_efficiency", DEFAULT_EFFICIENCY)
        self._energy_capacity = storage_params.get("energy_capacity", MAX_CAPACITY)
        self._power_capacity = storage_params.get("power_capacity", MAX_CAPACITY)
        self._init_time = storage_params.get("initial_time", INITIAL_TIME)

        init_state = StorageState(
            state_of_charge=self.state_of_charge,
            charging_efficiency=self.charging_efficiency,
            discharging_efficiency=self.discharging_efficiency,
            power_capacity=self.power_capacity,
            energy_capacity=self.energy_capacity,
        )

        super().__init__(storage_params, init_state=init_state)
        self.action_type = StorageAction

    @property
    def power_capacity(self) -> float:
        """
        float: Maximum amount of power the storage device can handle in [kW].
        """
        return self._power_capacity

    @power_capacity.setter
    def power_capacity(self, power_capacity: float) -> None:
        """
        Set the maximum power capacity of the battery.

        Parameters
        ----------
        power_capacity : float
            The new power capacity value in [kW]. If None, defaults to MAX_CAPACITY.

        Raises
        ------
        ValueError
            If `power_capacity` is less than `MIN_CAPACITY`.
        """
        power_capacity = MAX_CAPACITY if power_capacity is None else power_capacity
        if power_capacity < MIN_CAPACITY:
            raise ValueError(f"power_capacity must be >= {MIN_CAPACITY}.")
        self._power_capacity = power_capacity

    @property
    def energy_capacity(self) -> float:
        """
        float: Maximum amount of energy the storage device can store in [kWh].
        """
        return self._energy_capacity

    @energy_capacity.setter
    def energy_capacity(self, energy_capacity: float) -> None:
        """
        Set the maximum energy capacity of the battery.

        Parameters
        ----------
        energy_capacity : float
            The new energy capacity value in [kWh]. If None, defaults to MAX_CAPACITY.

        Raises
        ------
        ValueError
            If `energy_capacity` is less than `MIN_CAPACITY`.
        """
        energy_capacity = MAX_CAPACITY if energy_capacity is None else energy_capacity
        if energy_capacity < MIN_CAPACITY:
            raise ValueError(f"energy_capacity must be >= {MIN_CAPACITY}.")
        self._energy_capacity = energy_capacity

    @property
    def charging_efficiency(self) -> float:
        """
        float: Technical efficiency of the charging process.
        """
        return self._charging_efficiency

    @charging_efficiency.setter
    def charging_efficiency(self, charging_efficiency: float) -> None:
        """
        Set the charging efficiency of the battery.

        Parameters
        ----------
        charging_efficiency : float
            The new charging efficiency value (0 <= charging_efficiency <= 1). If None, defaults to DEFAULT_EFFICIENCY.

        Raises
        ------
        ValueError
            If `charging_efficiency` is not between `MIN_EFFICIENCY` and `MAX_EFFICIENCY`.
        """
        charging_efficiency = DEFAULT_EFFICIENCY if charging_efficiency is None else charging_efficiency
        if not (MIN_EFFICIENCY <= charging_efficiency <= MAX_EFFICIENCY):
            raise ValueError("charging_efficiency must be between 0 and 1.")
        self._charging_efficiency = charging_efficiency

    @property
    def discharging_efficiency(self) -> float:
        """
        float: Technical efficiency of the discharging process.
        """
        return self._discharging_efficiency

    @discharging_efficiency.setter
    def discharging_efficiency(self, discharging_efficiency: float) -> None:
        """
        Set the discharging efficiency of the battery.

        Parameters
        ----------
        discharging_efficiency : float
            The new discharging efficiency value (0 <= discharging_efficiency <= 1). If None, defaults to DEFAULT_EFFICIENCY.

        Raises
        ------
        ValueError
            If `discharging_efficiency` is not between `MIN_EFFICIENCY` and `MAX_EFFICIENCY`.
        """
        discharging_efficiency = DEFAULT_EFFICIENCY if discharging_efficiency is None else discharging_efficiency
        if not (MIN_EFFICIENCY <= discharging_efficiency <= MAX_EFFICIENCY):
            raise ValueError("discharging_efficiency must be between 0 and 1.")
        self._discharging_efficiency = discharging_efficiency

    @property
    def state_of_charge(self) -> float:
        """
        float: Current state of charge of the battery in [kWh].
        """
        return self._state_of_charge

    @state_of_charge.setter
    def state_of_charge(self, state_of_charge: float) -> None:
        """
        Set the current state of charge of the battery.

        Parameters
        ----------
        state_of_charge : float
            The new state of charge value in [kWh].

        Raises
        ------
        ValueError
            If `state_of_charge` is less than `MIN_CHARGE` or exceeds `energy_capacity`.
        """
        if state_of_charge < MIN_CHARGE:
            raise ValueError(f"state_of_charge must be >= {MIN_CHARGE}.")
        if state_of_charge > self.energy_capacity:
            raise ValueError("state_of_charge must be <= energy_capacity.")
        self._state_of_charge = state_of_charge

    def reset(self) -> None:
        """
        Reset the Battery to its initial state.

        This method restores the battery's properties to their initial values as defined in the initial state.
        """
        self._power_capacity = self.init_state.power_capacity
        self._energy_capacity = self.init_state.energy_capacity
        self._state_of_charge = self.init_state.state_of_charge
        self._charging_efficiency = self.init_state.charging_efficiency
        self._discharging_efficiency = self.init_state.discharging_efficiency
        super().reset()

    def update_state(self, state: StorageState) -> None:
        """
        Update the battery's state based on the provided `StorageState`.

        Parameters
        ----------
        state : StorageState
            The new state to update the battery with.
        """
        self.energy_capacity = state.energy_capacity
        self.power_capacity = state.power_capacity
        self.state_of_charge = state.state_of_charge
        self.charging_efficiency = state.charging_efficiency
        self.discharging_efficiency = state.discharging_efficiency

    def get_action_space(self) -> Bounds:
        """
        Define the action space for the battery.

        The action space is defined based on the current state of charge and the energy capacity.
        It allows actions to charge or discharge the battery within feasible limits.

        Returns
        -------
        Bounds
            The bounds of the action space, specifying the range of possible actions.
        """
        self.update_state(self.state)
        # Define lower and upper bounds for charging/discharging power
        # Negative values represent discharging, positive values represent charging
        low = -self.state_of_charge if self.state_of_charge > MIN_CHARGE else MIN_CHARGE
        high = self.energy_capacity - self.state_of_charge

        return Bounds(low=low, high=high, shape=(1,), dtype=np.float32)

    def get_observation_space(self) -> Bounds:
        """
        Define the observation space for the battery.

        The observation space includes time, state of charge, charging/discharging efficiencies,
        and power/energy capacities.

        Returns
        -------
        Bounds
            The bounds of the observation space, specifying the range of possible observations.
        """
        # Define the lower and upper bounds for each dimension of the observation space
        low = np.array(
            [INITIAL_TIME, MIN_CHARGE, MIN_EFFICIENCY, MIN_EFFICIENCY, MIN_CAPACITY, MIN_CAPACITY],
            dtype=np.float32,
        )  # Lower bounds for observations
        high = np.array(
            [
                MAX_TIME,
                self.energy_capacity,
                MAX_EFFICIENCY,
                MAX_EFFICIENCY,
                MAX_CAPACITY,
                MAX_CAPACITY,
            ],
            dtype=np.float32,
        )  # Upper bounds for observations
        return Bounds(low=low, high=high, shape=(len(low),), dtype=np.float32)

    def get_current_state(self) -> StorageState:
        """
        Retrieve the current state of the battery.

        Returns
        -------
        StorageState
            The current storage state of the battery.
        """
        return self.state

    def get_reward(self) -> float:
        """
        Calculate and return the reward for the current state.

        This method can be customized to define how rewards are calculated based on the battery's state.

        Returns
        -------
        float
            The reward value.
        """
        # Placeholder for reward calculation logic
        # This should be customized based on specific simulation requirements
        return 0.0

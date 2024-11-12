"""
Battery Dynamics Module for Smart Grid Simulation.

This module defines the BatteryDynamics class, which models the behavior of a battery
within the smart grid, handling charging and discharging actions, accounting for
efficiencies and natural decay losses.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from functools import partial

from ..energy_dynamcis import StorageDynamics  # Corrected import path
from ...model.state import StorageState
from ...model.action import StorageAction
from ...config import (
    MIN_CHARGE,
    MIN_EXPONENT,
    MAX_EXPONENT,
    DEFAULT_LIFETIME_CONSTANT,
)
from ...utils.utils import move_time_tick


class BatteryDynamics(StorageDynamics, ABC):
    """
    Battery Energy Dynamics.

    This class models the dynamics of a battery within the smart grid, handling charging
    and discharging actions, accounting for efficiencies and natural decay losses.
    """

    def __init__(self, dynamics_params: Optional[Any] = None) -> None:
        """
        Initialize the BatteryDynamics instance.

        Parameters
        ----------
        dynamics_params : Optional[Any], default=None
            Parameters defining the battery dynamics behavior. Currently not utilized,
            but can be extended for future enhancements.
        """
        super().__init__(dynamics_params)

    def do(
        self,
        action: Union[ArrayLike, StorageAction],
        state: Optional[StorageState] = None,
        **kwargs: Any
    ) -> StorageState:
        """
        Perform an action on the battery to update its state.

        This method handles charging and discharging actions, applies efficiencies,
        accounts for natural decay losses, and updates the battery's state.

        Parameters
        ----------
        action : Union[ArrayLike, StorageAction]
            The action to perform. Must be either a NumPy array with a single value
            or a `StorageAction` instance.
        state : Optional[StorageState], default=None
            The current state of the battery. If `None`, a default state must be provided.
        **kwargs : Any
            Additional keyword arguments, such as `lifetime_constant`.

        Returns
        -------
        StorageState
            The updated state of the battery after performing the action.

        Raises
        ------
        ValueError
            If the `action` is invalid or `state` is not provided.
        TypeError
            If the `action` is not of the expected type.
        """
        if state is None:
            raise ValueError("Battery state must be provided.")

        # Validate action input
        if isinstance(action, np.ndarray):
            if action.size != 1:
                raise ValueError("Action array must contain exactly one value.")
            action_value = StorageAction.from_numpy(action)
        elif isinstance(action, StorageAction):
            action_value = action
        else:
            raise TypeError("Action must be a numpy array or StorageAction instance.")

        charge_value = action_value.charge

        # Validate state parameters
        if state.energy_capacity < 0:
            raise ValueError("Energy capacity must be greater than zero.")
        if not (0 <= state.charging_efficiency <= 1):
            raise ValueError("Charging efficiency must be between 0 and 1.")
        if not (0 <= state.discharging_efficiency <= 1):
            raise ValueError("Discharging efficiency must be between 0 and 1.")

        # Apply charging or discharging efficiency
        if charge_value > 0:
            adjusted_value = charge_value * state.charging_efficiency
        else:
            adjusted_value = charge_value * state.discharging_efficiency

        # Apply natural decay losses
        lifetime_constant = kwargs.get('lifetime_constant', DEFAULT_LIFETIME_CONSTANT)

        if adjusted_value is not None:
            new_state = state.copy()

            # Update state of charge with bounds
            if adjusted_value > MIN_CHARGE:
                new_state.state_of_charge = min(
                    state.state_of_charge + adjusted_value, state.energy_capacity
                )
            else:
                new_state.state_of_charge = max(
                    state.state_of_charge + adjusted_value, MIN_CHARGE
                )

            # Update energy and power capacities with exponential decay
            new_state.energy_capacity = self.exp_mult(
                state.energy_capacity, state, lifetime_constant
            )
            new_state.power_capacity = self.exp_mult(
                state.power_capacity, state, lifetime_constant
            )

            # Efficiencies remain unchanged in this method
            new_state.charging_efficiency = state.charging_efficiency
            new_state.discharging_efficiency = state.discharging_efficiency

            # Advance time
            new_state.current_time_step, new_state.hour = move_time_tick(
                state.current_time_step, state.hour
            )

            return new_state
        else:
            raise ValueError("Invalid action value.")

    def predict(
        self,
        action: Union[ArrayLike, StorageAction],
        state: Optional[StorageState] = None,
        **kwargs: Any
    ) -> StorageState:
        """
        Predict the next state of the battery given an action.

        This method simulates the battery's response to an action without mutating the original state.

        Parameters
        ----------
        action : Union[ArrayLike, StorageAction]
            The action to perform. Must be either a NumPy array with a single value
            or a `StorageAction` instance.
        state : Optional[StorageState], default=None
            The current state of the battery. If `None`, a default state must be provided.
        **kwargs : Any
            Additional keyword arguments, such as `lifetime_constant`.

        Returns
        -------
        StorageState
            The predicted state of the battery after performing the action.

        Raises
        ------
        ValueError
            If the `action` is invalid or `state` is not provided.
        TypeError
            If the `action` is not of the expected type.
        """
        if state is None:
            raise ValueError("Battery state must be provided for prediction.")

        # Validate action input
        if isinstance(action, np.ndarray):
            if action.size != 1:
                raise ValueError("Action array must contain exactly one value.")
            action_value = StorageAction.from_numpy(action)
        elif isinstance(action, StorageAction):
            action_value = action
        else:
            raise TypeError("Action must be a numpy array or StorageAction instance.")

        charge_value = action_value.charge

        # Validate state parameters
        if state.energy_capacity < 0:
            raise ValueError("Energy capacity must be greater than zero.")
        if not (0 <= state.charging_efficiency <= 1):
            raise ValueError("Charging efficiency must be between 0 and 1.")
        if not (0 <= state.discharging_efficiency <= 1):
            raise ValueError("Discharging efficiency must be between 0 and 1.")

        # Apply charging or discharging efficiency
        if charge_value > 0:
            adjusted_value = charge_value * state.charging_efficiency
        else:
            adjusted_value = charge_value * state.discharging_efficiency

        # Apply natural decay losses
        lifetime_constant = kwargs.get('lifetime_constant', DEFAULT_LIFETIME_CONSTANT)

        if adjusted_value is not None:
            predicted_state = state.copy()

            # Update state of charge with bounds
            if adjusted_value > MIN_CHARGE:
                predicted_state.state_of_charge = min(
                    state.state_of_charge + adjusted_value, state.energy_capacity
                )
            else:
                predicted_state.state_of_charge = max(
                    state.state_of_charge + adjusted_value, MIN_CHARGE
                )

            # Update energy and power capacities with exponential decay
            predicted_state.energy_capacity = self.exp_mult(
                state.energy_capacity, state, lifetime_constant
            )
            predicted_state.power_capacity = self.exp_mult(
                state.power_capacity, state, lifetime_constant
            )

            # Efficiencies remain unchanged in this method
            predicted_state.charging_efficiency = state.charging_efficiency
            predicted_state.discharging_efficiency = state.discharging_efficiency

            # Advance time
            predicted_state.current_time_step, predicted_state.hour = move_time_tick(
                state.current_time_step, state.hour
            )

            return predicted_state
        else:
            raise ValueError("Invalid action value.")

    def get_current_charge_capability(self, state: StorageState) -> float:
        """
        Get the current charge capability of the battery.

        This method calculates how much energy the battery can currently accept for charging
        based on its current state.

        Parameters
        ----------
        state : StorageState
            The current state of the battery.

        Returns
        -------
        float
            The current charge capability in [kWh].
        """
        available_capacity = state.energy_capacity - state.state_of_charge
        return max(0.0, available_capacity)

    def get_current_discharge_capability(self, state: StorageState) -> float:
        """
        Get the current discharge capability of the battery.

        This method calculates how much energy the battery can currently provide for discharging
        based on its current state.

        Parameters
        ----------
        state : StorageState
            The current state of the battery.

        Returns
        -------
        float
            The current discharge capability in [kWh].
        """
        return max(MIN_CHARGE, state.state_of_charge)

    def predict_charge_capability(self, state: StorageState) -> float:
        """
        Predict the future charge capability of the battery based on the current state.

        This method can incorporate factors like expected efficiency changes or capacity decay.

        Parameters
        ----------
        state : StorageState
            The current state of the battery.

        Returns
        -------
        float
            The predicted charge capability in [kWh].
        """
        # For simplicity, assume charge capability remains the same
        # Extend this method with more sophisticated predictions as needed
        return self.get_current_charge_capability(state)

    def predict_discharge_capability(self, state: StorageState) -> float:
        """
        Predict the future discharge capability of the battery based on the current state.

        This method can incorporate factors like expected efficiency changes or capacity decay.

        Parameters
        ----------
        state : StorageState
            The current state of the battery.

        Returns
        -------
        float
            The predicted discharge capability in [kWh].
        """
        # For simplicity, assume discharge capability remains the same
        # Extend this method with more sophisticated predictions as needed
        return self.get_current_discharge_capability(state)

    @staticmethod
    def exp_mult(x: float, state: StorageState, lifetime_constant: float) -> float:
        """
        Apply exponential decay to a value based on the lifetime constant and current time step.

        This function ensures that the exponent is clamped within a safe range to prevent overflow.

        Parameters
        ----------
        x : float
            The original value to be decayed.
        state : StorageState
            The current state of the battery, used to determine the current time step.
        lifetime_constant : float
            The lifetime constant representing the rate of decay.

        Returns
        -------
        float
            The decayed value.

        Raises
        ------
        ValueError
            If `lifetime_constant` is negative.
        """
        if lifetime_constant < 0:
            raise ValueError("Lifetime constant must be non-negative.")

        if lifetime_constant == 0:
            return x  # No decay if lifetime constant is zero

        # Calculate the exponent and clamp it to prevent overflow
        exponent = state.current_time_step / float(lifetime_constant)
        exponent = max(MIN_EXPONENT, min(MAX_EXPONENT, exponent))

        return x * np.exp(-exponent)
